"""Stream only pinned SigLIP2 Giant vision tensors into CUDA, without a CPU tower."""
from pathlib import Path
import numpy as np
from common.config import digest, read_json
from common.local_paths import local_path

REPO = 'google/siglip2-giant-opt-patch16-384'
REVISION = 'a713301b217d38485fb2204c808367d10bc3cc40'


def make_encoder(manifest):
    import torch
    from transformers import SiglipVisionModel, SiglipVisionConfig, SiglipImageProcessor
    from safetensors import safe_open
    if manifest['model'] != REPO or manifest['revision'] != REVISION or digest(__file__) != manifest['loader_source_sha256']:
        raise ValueError('Only the pinned Giant vision protocol is supported')
    folder = local_path('cache_root') / 'siglip2_giant_weights' / REVISION
    provenance = read_json(folder / 'download.json')
    if provenance['repo'] != REPO or provenance['revision'] != REVISION or not provenance.get('sharded'):
        raise ValueError('Official Giant shard provenance is missing')
    index_path = folder / 'model.safetensors.index.json'
    if digest(index_path) != manifest['checkpoint_index_sha256']:
        raise ValueError('Pinned Giant tensor index changed')
    weight_map = {name: shard for name, shard in read_json(index_path)['weight_map'].items() if name.startswith('vision_model.')}
    if sorted(set(weight_map.values())) != provenance['vision_weight_shards']:
        raise ValueError('Giant vision shard coverage differs')
    for shard in provenance['vision_weight_shards']:
        record = provenance['files'].get(shard)
        if record is None or not record['publisher_lfs_verified'] or record['sha256'] != manifest['checkpoint_shard_sha256'][shard] or digest(folder / shard) != record['sha256']:
            raise ValueError('Giant shard checksum is not verified')
    config = read_json(folder / 'config.json')
    if config.get('model_type') != 'siglip':
        raise ValueError('Unexpected Giant model type')
    with torch.device('meta'):
        visual = SiglipVisionModel(SiglipVisionConfig(**config['vision_config']))
    expected = {name: tuple(value.shape) for name, value in visual.state_dict().items()}
    if set(expected) != set(weight_map):
        raise ValueError('Giant vision tensor names do not match the architecture')
    visual.requires_grad_(False)
    selected = {}
    for shard in provenance['vision_weight_shards']:
        with safe_open(folder / shard, framework='pt', device='cpu') as tensors:
            for name in sorted(key for key, value in weight_map.items() if value == shard):
                if tuple(tensors.get_slice(name).get_shape()) != expected[name]:
                    raise ValueError('Giant vision tensor shape mismatch')
                # At most one host tensor is staged; unused text tensors are never read.
                tensor = tensors.get_tensor(name)
                selected[name] = tensor.to(device='cuda', dtype=torch.float16)
                del tensor
    visual.load_state_dict(selected, strict=True, assign=True)
    del selected
    embeddings = visual.vision_model.embeddings
    embeddings.position_ids = torch.arange(embeddings.num_positions, device='cuda').expand((1, -1))
    if any(value.is_meta for value in list(visual.parameters()) + list(visual.buffers())):
        raise ValueError('Giant has an unmaterialized Meta tensor')
    visual.eval()
    processor = SiglipImageProcessor.from_pretrained(folder, local_files_only=True)
    if processor.to_dict() != manifest['preprocess']:
        raise ValueError('Giant official preprocessing changed')
    def encode(images):
        pixels = processor(images=images, return_tensors='pt')['pixel_values'].half().cuda()
        with torch.inference_mode():
            values = visual(pixel_values=pixels).pooler_output.float()
            values = (values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().numpy().astype(np.float64)
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return encode
