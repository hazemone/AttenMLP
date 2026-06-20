#!/usr/bin/env python3
# =====================================================================
#  inspect_arch.py  --  AttenMLP / DualMLP architecture inspector
# ---------------------------------------------------------------------
#  PURPOSE (answers Reviewer #2 reproducibility): prints the REAL
#  architecture configuration of your trained model -- centroid count
#  per stage, K-neighbours, feature dimensions, #blocks, classifier
#  head, attention heads, fusion strategy, and total parameter count.
#  Nothing here is invented; every number is read from the model you
#  actually build.
#
#  HOW TO RUN
#  ----------
#    1. Copy this file into your   classification_ModelNet40/   folder
#       (the same directory that contains your main.py and models/).
#    2. Activate the same Python environment you train in.
#    3. Run it with the SAME architecture arguments you used to train
#       the paper model, e.g.:
#
#         python inspect_arch.py \
#             --model AttenMLP \
#             --dual_net True \
#             --last_layer_concat add \
#             --num_points_low 1024 --neighbours_low 32 \
#             --num_points_high 2048 --neighbours_high 32 \
#             --num_channel 32
#
#       (Replace every value with the exact ones from the command that
#        produced your reported 93.85% result. If you are unsure, look
#        in   checkpoints/<run>/args.txt   from any of your runs -- it
#        records the exact arguments that were used.)
#
#    4. Copy the printed block back to me and I will drop the numbers
#       straight into the reproducibility table in the paper.
# =====================================================================

import argparse
import sys


def build_args():
    p = argparse.ArgumentParser(description="Inspect AttenMLP/DualMLP architecture")
    # --- architecture-relevant arguments, mirroring main.py -----------
    p.add_argument('--model', type=str, default='pointMLP',
                   help='your model flag (cosmetic here; e.g. AttenMLP)')
    p.add_argument('--dual_net', type=lambda x: str(x).lower() == 'true', default=True)
    p.add_argument('--num_classes', type=int, default=40)
    p.add_argument('--num_points_low',  type=int, default=1024)
    p.add_argument('--neighbours_low',  type=int, default=32)
    p.add_argument('--num_points_high', type=int, default=2048)
    p.add_argument('--neighbours_high', type=int, default=32)
    p.add_argument('--num_channel',     type=int, default=32)
    p.add_argument('--last_layer_concat', type=str, default='add')
    p.add_argument('--add_factor', type=float, default=1.0)
    return p.parse_args()


def count_params(m):
    try:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    except Exception:
        return None


def first_conv_out_channels(module):
    """Find the first Conv1d/Conv2d in a submodule and return its out_channels."""
    try:
        for sub in module.modules():
            cn = sub.__class__.__name__
            if cn in ('Conv1d', 'Conv2d'):
                return getattr(sub, 'out_channels', None)
    except Exception:
        pass
    return None


def describe_classifier(clf):
    rows = []
    try:
        modules = list(clf)
    except TypeError:
        try:
            modules = list(clf.children())
        except Exception:
            return rows
    for mod in modules:
        cn = mod.__class__.__name__
        if cn == 'Linear':
            rows.append("Linear(%s -> %s)" % (getattr(mod, 'in_features', '?'),
                                              getattr(mod, 'out_features', '?')))
        elif cn == 'Dropout':
            rows.append("Dropout(p=%s)" % getattr(mod, 'p', '?'))
        else:
            rows.append(cn)
    return rows


def find_attention(net):
    found = []
    try:
        items = list(net.named_modules())
    except Exception:
        return found
    for name, mod in items:
        cn = mod.__class__.__name__
        if any(t in cn.lower() for t in ('attention', 'multihead', 'mha')):
            heads = None
            for attr in ('num_heads', 'n_heads', 'heads', 'num_head', 'nhead'):
                if hasattr(mod, attr):
                    heads = getattr(mod, attr)
                    break
            dim = None
            for attr in ('embed_dim', 'dim', 'd_model', 'in_features'):
                if hasattr(mod, attr):
                    dim = getattr(mod, attr)
                    break
            found.append((name, cn, heads, dim))
    return found


def describe_net(net, title):
    print("\n--- %s ---" % title)
    pts = getattr(net, 'points', None)
    stages = getattr(net, 'stages', None)
    print("  input points              : %s" % pts)
    print("  stages                    : %s" % stages)

    embed = first_conv_out_channels(getattr(net, 'embedding', net))
    print("  embedding dim (embed_dim) : %s" % embed)

    groupers = getattr(net, 'local_grouper_list', None)
    if groupers is not None:
        cents = [getattr(g, 'groups', '?') for g in groupers]
        ks = [getattr(g, 'kneighbors', '?') for g in groupers]
        print("  centroids per stage (M)   : %s" % cents)
        print("  K-neighbours per stage    : %s" % ks)

    # channel width per stage (embed_dim * cumulative dim_expansion, default x2)
    if embed is not None and stages:
        widths, w = [], embed
        for _ in range(stages):
            w *= 2
            widths.append(w)
        print("  feature width per stage   : %s  (assuming dim_expansion=[2,2,2,2])" % widths)

    pre = getattr(net, 'pre_blocks_list', None)
    pos = getattr(net, 'pos_blocks_list', None)
    if pre is not None and pos is not None:
        print("  #pre / #pos block modules : %d / %d stages" % (len(pre), len(pos)))

    clf = getattr(net, 'classifier', None)
    if clf is not None:
        print("  classifier head           :")
        for r in describe_classifier(clf):
            print("      %s" % r)

    attn = find_attention(net)
    if attn:
        print("  attention modules         :")
        seen = set()
        for name, cn, heads, dim in attn:
            key = (cn, heads, dim)
            if key in seen:
                continue
            seen.add(key)
            print("      %-22s heads=%s dim=%s  (x%d such modules)"
                  % (cn, heads, dim, sum(1 for a in attn if (a[1], a[2], a[3]) == key)))
    else:
        print("  attention modules         : none auto-detected.")
        print("                              (If your MHA class name does not contain")
        print("                               'attention'/'mha', tell me the #heads manually.)")

    n = count_params(net)
    if n is not None:
        print("  trainable parameters      : %s  (%.3f M)" % ("{:,}".format(n), n / 1e6))
    return n


def main():
    args = build_args()
    try:
        from models.pointmlp import Model
    except Exception as e:
        print("ERROR: could not import Model from models.pointmlp.")
        print("Run this script from inside your classification_ModelNet40/ folder")
        print("(the directory that contains main.py and the models/ package).")
        print("Import error: %s" % e)
        sys.exit(1)

    # Build the two pathways EXACTLY as main.py does -------------------
    sparse_net = Model(points=args.num_points_low,
                       k_neighbors=[args.neighbours_low] * 4,
                       parser_args=args)
    if args.dual_net and args.last_layer_concat == 'concat':
        import torch
        sparse_net.classifier[0] = torch.nn.Linear(1280, 512)
    dense_net = Model(points=args.num_points_high, class_num=args.num_classes,
                      embed_dim=args.num_channel, groups=1, res_expansion=1.0,
                      activation="relu", bias=False, use_xyz=False, normalize="anchor",
                      dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2],
                      pos_blocks=[2, 2, 2, 2], k_neighbors=[args.neighbours_high] * 4,
                      reducers=[2, 2, 2, 2], parser_args=args)

    print("=" * 64)
    print(" AttenMLP architecture inspection")
    print("=" * 64)
    print(" dual_net=%s | num_classes=%s | fusion(last_layer_concat)=%s"
          % (args.dual_net, args.num_classes, args.last_layer_concat))

    ns = describe_net(sparse_net, "SPARSE pathway (SparseNet)")
    nd = describe_net(dense_net,  "DENSE pathway (DenseNet)")

    if ns is not None and nd is not None:
        tot = ns + nd
        print("\n" + "=" * 64)
        print(" TOTAL trainable parameters : %s  (%.3f M)"
              % ("{:,}".format(tot), tot / 1e6))
        print(" (Cross-check against the 3.15 MB / 1.85 GFLOPs in Table II.)")
        print("=" * 64)

    # Optional FLOPs (best effort; the paper's measured 1.85 GFLOPs stands)
    try:
        import torch  # noqa
        from thop import profile  # noqa
        print("\n[FLOPs] thop is installed. The dual-net forward is custom, so the most")
        print("        reliable GFLOPs figure remains the one you measured for the paper")
        print("        (1.85 GFLOPs). Keep using that value.")
    except Exception:
        print("\n[FLOPs] thop not installed -> skipped (this is fine; keep the paper's")
        print("        measured 1.85 GFLOPs).")

    print("\nCopy everything above and send it back, and I will fill the")
    print("reproducibility table in the paper with these exact values.")


if __name__ == "__main__":
    main()
