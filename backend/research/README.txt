Put your UNMODIFIED research files here (same folder as model6.py):
  extra_feats.py   nli_features.py   synth.py      <- model6.py imports synth at module level even though synth_ratio=0
model6.py is already here (byte-identical copy of the one you uploaded). The backend imports Net/collate/predict/tok_feats from it; the architecture is not duplicated.
