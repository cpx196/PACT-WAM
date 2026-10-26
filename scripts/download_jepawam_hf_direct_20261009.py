"""Download the Qwen JEPA-WAM assets directly, without proxy environment use."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import quote

import requests

def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def valid(path, item):
    if not path.is_file() or path.stat().st_size != item["size"]:
        return False
    if item.get("sha256"):
        return digest(path) == item["sha256"]
    raw = path.read_bytes()
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == item["blobId"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    session = requests.Session()
    session.trust_env = False
    plan = []
    for repo in ("Qwen/Qwen2.5-0.5B", "CokeAnd1ce/JEPA_WAM"):
        response = session.get(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=(15, 120))
        response.raise_for_status()
        info = response.json()
        for entry in info["siblings"]:
            name = entry["rfilename"]
            if repo.startswith("Qwen/"):
                chosen = "/" not in name and (name.endswith(".safetensors") or name in {
                    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                    "vocab.json", "merges.txt", "special_tokens_map.json", "model.safetensors.index.json",
                    "LICENSE", "README.md"})
                relative = "Qwen2.5-0.5B/" + name
            else:
                chosen = name.startswith(("checkpoints/libero/", "checkpoints/pretrained_vlm/")) and (
                    name.endswith("latest-checkpoint.pt") or name.endswith("step-040000-epoch-37-loss=0.0262.pt")
                    or name.endswith(("config.json", "dataset_statistics.json", "run-metrics.jsonl")))
                relative = "JEPA_WAM/" + name
            if not chosen:
                continue
            plan.append({"repo": repo, "revision": info["sha"], "source_path": name,
                "path": relative, "size": entry["size"], "blobId": entry["blobId"],
                "sha256": entry.get("lfs", {}).get("sha256")})
    plan.sort(key=lambda item: item["size"])
    (root / "hf_manifest.json").write_text(json.dumps(plan, indent=2))
    print(f"DIRECT plan: {len(plan)} files, {sum(x['size'] for x in plan)/1e9:.3f} GB", flush=True)
    for item in plan:
        target = root / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if valid(target, item):
            print("VERIFIED existing", item["path"], flush=True)
            continue
        if target.exists():
            raise RuntimeError(f"Existing final file failed verification; preserving it: {target}")
        partial = target.with_name(target.name + ".part")
        url = f"https://huggingface.co/{item['repo']}/resolve/{item['revision']}/{quote(item['source_path'], safe='/')}?download=true"
        for attempt in range(1, 6):
            try:
                if not valid(partial, item):
                    offset = partial.stat().st_size if partial.exists() else 0
                    headers = {"Range": f"bytes={offset}-"} if offset else {}
                    with session.get(url, headers=headers, stream=True, timeout=(20, 120)) as download:
                        download.raise_for_status()
                        if offset and download.status_code != 206:
                            raise RuntimeError("Server ignored resume Range; preserving partial file")
                        if offset and not download.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise RuntimeError("Incorrect resume Content-Range")
                        print("DOWNLOAD", item["path"], "offset", offset, flush=True)
                        last_report = start = time.monotonic()
                        received = 0
                        with partial.open("ab" if offset else "wb") as stream:
                            for chunk in download.iter_content(4 * 1024 * 1024):
                                if chunk:
                                    stream.write(chunk)
                                    received += len(chunk)
                                now = time.monotonic()
                                if now - last_report >= 15:
                                    print(f"PROGRESS {item['path']} {(offset+received)/item['size']:.1%} {received/(now-start)/1e6:.2f} MB/s", flush=True)
                                    last_report = now
                if not valid(partial, item):
                    raise RuntimeError("Downloaded size/hash did not match manifest")
                partial.rename(target)
                print("VERIFIED", item["path"], flush=True)
                break
            except Exception as error:
                print("RETRY", attempt, item["path"], repr(error), flush=True)
                if attempt == 5:
                    raise
                time.sleep(10)
    (root / "hf_download_complete.json").write_text(json.dumps({"files": len(plan), "bytes": sum(x["size"] for x in plan), "verified": True}, indent=2))
    print("ALL HF ASSETS VERIFIED", flush=True)

if __name__ == "__main__":
    main()
