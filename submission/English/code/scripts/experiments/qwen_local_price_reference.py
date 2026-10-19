"""Independent Qwen zero-shot price evaluation. Never exports a Kaggle CSV."""
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import re
import time
import uuid
from pathlib import Path

from common.config import digest, write_json
from common.local_paths import local_path, REPOSITORY
from download_qwen_reference import download

PROMPT = ('Estimate the sale price of the property shown in this image. '
          'The dataset contains Southern California real-estate listing photographs from around 2019. '
          'Use only the visible image and this dataset-level context. Do not search or identify the listing. '
          'Express the price in THOUSANDS of US dollars: for example, 750 means $750,000. '
          'Return exactly one JSON object with one numeric field: {"price": 750}. '
          'Do not return explanations, a range, dollar symbols, or other fields.')


def parse_price(text):
    cleaned = text.strip()
    if cleaned.startswith('```'):
        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned).strip()
    try:
        data = json.loads(cleaned)
        value = data['price'] if isinstance(data, dict) and set(data) == {'price'} else data if isinstance(data, (int, float)) else None
    except (ValueError, KeyError, TypeError):
        value = float(cleaned) if re.fullmatch(r'[+]?(?:\d+(?:\.\d*)?|\.\d+)', cleaned) else None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def wait_predecessor(path):
    """Wait on the real Windows handle, then verify terminal success."""
    before = json.loads(path.read_text(encoding='utf-8'))
    pid = before.get('pid')
    if not pid:
        raise ValueError('Predecessor lacks a recorded process ID')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.WaitForSingleObject.restype = ctypes.c_ulong
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x100000, False, pid)
    if handle:
        try:
            print(f'Qwen waiting for actual predecessor PID {pid}', flush=True)
            while True:
                value = kernel.WaitForSingleObject(handle, 1000)
                if value == 0:
                    break
                if value != 258:
                    raise OSError(ctypes.get_last_error(), 'Process wait failed')
        finally:
            kernel.CloseHandle(handle)
    after = json.loads(path.read_text(encoding='utf-8'))
    if after['status'] != 'completed':
        raise RuntimeError('Predecessor not completed; no automatic restart')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='Qwen/Qwen3.5-2B')
    parser.add_argument('--weights-dir', type=Path)
    parser.add_argument('--predecessor-status', type=Path)
    args = parser.parse_args()
    name = args.repo.split('/')[-1] + '_zero_shot'
    run = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
    relative = Path('18_mllm_local_only') / name / run
    public = REPOSITORY / 'output/experiment_record' / relative
    out = local_path('records_root') / relative
    public.mkdir(parents=True, exist_ok=False)
    out.mkdir(parents=True, exist_ok=False)
    registration = {'repo': args.repo, 'course_eligible': False, 'independent_local_only': True,
                    'included_in_50000_course_goal': False, 'test_images_used': False,
                    'kaggle_export': False, 'prompt': PROMPT, 'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
                    'split_sha256': digest(local_path('split_path')), 'source_sha256': digest(__file__),
                    'prediction_unit': 'thousand USD', 'max_new_tokens': 48, 'do_sample': False,
                    'enable_thinking': False, 'dtype': 'float16', 'batch_size': 1,
                    'max_pixels': 196608, 'min_pixels': 3136,
                    'invalid_output_policy': 'Training partition mean price; all validation rows remain scored',
                    'selection': 'Single preregistered prompt; no validation-selected prompt or calibration',
                    'authorization': 'User explicitly selected independent local validation on 2026-10-14'}
    write_json(public / 'registration.json', registration)
    def status(state, **details):
        write_json(public / 'status.json', {'status': state, 'pid': os.getpid(), **details})
    start = time.perf_counter()
    try:
        status('downloading')
        weights = args.weights_dir or download(args.repo)
        provenance = json.loads((weights / 'download.json').read_text(encoding='utf-8'))
        if provenance['repo'] != args.repo:
            raise ValueError('Checkpoint repository differs from registered model')
        for filename, record in provenance['files'].items():
            if digest(weights / filename) != record['sha256']:
                raise ValueError(f'Checkpoint file changed: {filename}')
            if filename.endswith('.safetensors') and not record['publisher_lfs_verified']:
                raise ValueError('Model weights were not verified against publisher checksum')
        registration.update(revision=provenance['revision'], checkpoint_files=provenance['files'])
        write_json(public / 'registration.json', registration)
        if args.predecessor_status:
            status('waiting', predecessor_status=str(args.predecessor_status))
            wait_predecessor(args.predecessor_status)
        import numpy as np
        import pandas as pd
        import torch
        import transformers
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from PIL import Image, ImageOps
        from common.metrics import save_predictions
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for this registered experiment')
        frame = pd.read_csv(local_path('split_path'), dtype={'imageid': str, 'group_id': str})
        train = frame[frame.partition.eq('train')].copy()
        valid = frame[frame.partition.eq('validation')].copy()
        if len(train) != 6399 or len(valid) != 1601 or set(train.group_id) & set(valid.group_id):
            raise ValueError('Unexpected frozen group split')
        registration['environment'] = {name: importlib.metadata.version(name) for name in
                                       ['torch', 'transformers', 'accelerate', 'huggingface-hub']}
        registration['gpu'] = torch.cuda.get_device_name(0)
        registration['split_counts'] = {'train': len(train), 'validation': len(valid)}
        fallback = float(train.price.mean())
        registration['invalid_fallback_price'] = fallback
        write_json(public / 'registration.json', registration)
        status('loading')
        processor = AutoProcessor.from_pretrained(weights, local_files_only=True, trust_remote_code=False,
                                                   min_pixels=3136, max_pixels=196608)
        # New processors use size rather than the older min/max_pixels constructor keywords.
        processor.image_processor.size = {'shortest_edge': 3136, 'longest_edge': 196608}
        registration['image_processor'] = processor.image_processor.to_dict()
        write_json(public / 'registration.json', registration)
        model = AutoModelForImageTextToText.from_pretrained(weights, local_files_only=True, trust_remote_code=False,
                        dtype=torch.float16, device_map='cuda', attn_implementation='sdpa')
        model.eval().requires_grad_(False)
        # Do not transmit an image ID, group ID, individual price or listing metadata to the model.
        def infer(imageid):
            with Image.open(local_path('data_root') / 'train' / imageid) as source:
                im = ImageOps.exif_transpose(source).convert('RGB')
            messages = [{'role': 'user', 'content': [{'type': 'image', 'image': im},
                                                    {'type': 'text', 'text': PROMPT}]}]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                         enable_thinking=False, return_dict=True, return_tensors='pt').to('cuda')
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].is_floating_point():
                    inputs[key] = inputs[key].to(torch.float16)
            begun = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=48)
            answer = processor.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return {'imageid': imageid, 'raw_response': answer, 'parsed_price': parse_price(answer),
                    'seconds': time.perf_counter() - begun,
                    'input_tokens': int(inputs['input_ids'].shape[1])}
        # Preflight uses training images only; it checks execution, not model/prompt selection.
        ordered = train.sort_values(['price', 'imageid']).reset_index(drop=True)
        probes = ordered.iloc[np.linspace(0, len(ordered) - 1, 12).round().astype(int)].copy()
        status('preflight')
        preflight = []
        for imageid in probes.imageid:
            row = infer(imageid)
            preflight.append(row)
            print(f'Qwen preflight {len(preflight)}/12 price={row["parsed_price"]} seconds={row["seconds"]:.2f}', flush=True)
            write_json(out / 'preflight_responses.json', preflight)
        write_json(public / 'preflight.json', {'n': 12, 'parsed_n': sum(r['parsed_price'] is not None for r in preflight),
                   'mean_seconds': float(np.mean([r['seconds'] for r in preflight])),
                   'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2})
        if not any(r['parsed_price'] is not None for r in preflight):
            raise RuntimeError('All training preflight outputs invalid; do not start expensive validation')
        status('evaluating', images=0, total=len(valid))
        responses = []
        with (out / 'validation_responses.jsonl').open('x', encoding='utf-8') as stream:
            for imageid in valid.imageid:
                row = infer(imageid)
                responses.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                if len(responses) % 20 == 0 or len(responses) == len(valid):
                    status('evaluating', images=len(responses), total=len(valid))
                    print(f'Qwen local validation {len(responses)}/{len(valid)}', flush=True)
        raw = np.array([r['parsed_price'] if r['parsed_price'] is not None else fallback for r in responses])
        result = {'head': 'zero_shot_direct_price', **save_predictions(out, valid, raw, 1300),
                  'course_eligible': False, 'independent_local_only': True,
                  'invalid_n': sum(r['parsed_price'] is None for r in responses),
                  'mean_inference_seconds': float(np.mean([r['seconds'] for r in responses])),
                  'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
                  'seconds': time.perf_counter() - start}
        slices = []
        for tier, mask in [('low', valid.price.to_numpy() <= 300),
                           ('middle', (valid.price.to_numpy() > 300) & (valid.price.to_numpy() <= 1300)),
                           ('high', valid.price.to_numpy() > 1300)]:
            error = raw[mask] - valid.price.to_numpy()[mask]
            slices.append({'tier': tier, 'n': int(mask.sum()), 'mse': float(np.mean(error**2)),
                           'mae': float(np.mean(abs(error))), 'bias': float(np.mean(error))})
        write_json(public / 'results.json', result)
        write_json(public / 'price_slices.json', slices)
        # This aggregate report deliberately cannot be mistaken for a eligible Kaggle artifact.
        report = ('# Qwen 独立本地效果对照\n\n用户已选择独立本地验证；预训练MLLM违反课程方法限制。'
                  '本实验不计入合规MSE<50000目标，不生成测试集或Kaggle提交文件。\n\n'
                  f'固定验证集1601张，原价MSE {result["mse"]:.2f}，MAE {result["mae"]:.2f}，'
                  f'高价MSE {result["tail_mse"]:.2f}，无效输出 {result["invalid_n"]} 张。'
                  '无效输出以训练分区平均价格回退，保留全部验证样本评价。\n\n'
                  '提示词、模型版本、权重校验、输入分辨率和推理配置在registration.json；'
                  '逐图原始回答与预测保存在.local。不向模型输入单图标签、编号或来源信息，'
                  '不搜索房源、不查询测试真值。预训练数据污染无法通过该本地对照排除。\n')
        (public / '实验报告.md').write_text(report, encoding='utf-8')
        status('completed', mse=result['mse'], images=len(valid), total=len(valid))
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
