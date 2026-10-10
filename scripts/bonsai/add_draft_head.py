# Copy a Bonsai 2 MTP GGUF and add blk.<mtp>.nextn.draft_head (rows of output.weight for the top-K frequent tokens + all
# special tokens) and draft_head_ids (I32), and list the head under prism.hadamard.weight_names (same input rotation).
# usage: add_draft_head.py in.gguf out.gguf freq.json K
import json, sys, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
import gguf
src, dst, freq, K = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
r = gguf.GGUFReader(src)
arch = r.fields["general.architecture"].contents()
n_layer = r.fields[f"{arch}.block_count"].contents()
mtp = n_layer - r.fields[f"{arch}.nextn_predict_layers"].contents()
ttype = np.array(r.fields["tokenizer.ggml.token_type"].contents())
special = [int(i) for i in np.nonzero(ttype != 1)[0]]
cnt = {int(k): v for k, v in json.load(open(freq)).items()}
ranked = [t for t, _ in sorted(cnt.items(), key=lambda kv: -kv[1]) if ttype[t] == 1]
ids = special + [t for t in ranked if t not in set(special)]
ids = ids[:K]
if len(ids) < K:  # pad with unseen normal tokens in id order (rare)
    seen = set(ids); ids += [t for t in range(len(ttype)) if t not in seen][:K - len(ids)]
ids = np.array(ids, dtype=np.int32)
print(f"mtp block {mtp}, K {K}: {len(special)} special + {K - len(special)} frequent", flush=True)

w = gguf.GGUFWriter(dst, arch, use_temp_file=False)
for f in r.fields.values():
    if f.name.startswith("GGUF.") or f.name == "general.architecture":
        continue
    vt = f.types[0]
    v = f.contents()
    if f.name == "prism.hadamard.weight_names":
        v = list(v) + [f"blk.{mtp}.nextn.draft_head.weight"]
    if vt == gguf.GGUFValueType.ARRAY:
        w.add_key_value(f.name, v, vt, sub_type=f.types[-1] if len(f.types) > 1 else None)
    else:
        w.add_key_value(f.name, v, vt)
out = next(t for t in r.tensors if t.name == "output.weight")
rows = np.ascontiguousarray(np.asarray(out.data)[ids])      # [K, row bytes]
for t in r.tensors:
    w.add_tensor(t.name, np.asarray(t.data), raw_shape=list(np.asarray(t.data).shape), raw_dtype=t.tensor_type)
w.add_tensor(f"blk.{mtp}.nextn.draft_head.weight", rows, raw_shape=list(rows.shape), raw_dtype=out.tensor_type)
w.add_tensor(f"blk.{mtp}.nextn.draft_head_ids.weight", ids)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(progress=False); w.close()
print("written", dst, flush=True)
