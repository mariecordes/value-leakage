"""Shared colours for every figure. One place to change them.

Chosen over two alternatives, both checked with a contrast/colour-blindness
validator:
- The paper's blue/purple fails: the two lines its result rests on are dE 2.4
  apart under red-blindness (and 11.5 for normal vision, against a floor of 15).
- The starter repo's blue/orange passes but is default-matplotlib blue.

Indigo and amber keep the repo's meaning — cool = below pays, warm = above pays —
with deeper hues and the best separation of any pair tested (dE 29.0 under
red-blindness, 33.6 normal). Baseline stays a neutral grey: it is a reference,
not a result, and it is drawn dashed so it never depends on colour alone.
"""

BASELINE = "#8a9199"   # no bet — reference, drawn dashed
BELOW    = "#3d4fa8"   # indigo — below the threshold pays
ABOVE    = "#c77a00"   # amber  — above the threshold pays

INDIGO_LT = "#7e93d6"  # lighter indigo, for the four-label breakdown
WHITE     = "#ffffff"

# Keys match the repo's condition names so this drops into existing code.
COLORS = {"baseline": BASELINE, "below_good": BELOW, "above_good": ABOVE}

# Four disclosure labels as one openness scale: open about it -> silent ->
# denies it. "Never raises it" is white with a hatch rather than a fourth hue —
# no light neutral separates cleanly from a light indigo, and it is 4.5% of the
# data and our least reliable label, so marking it as visually different is
# honest.
AMBER_LT = "#f2c98a"
DISCLOSURE = {"ADMITS": BELOW, "MENTIONS": INDIGO_LT,
              "NO_MENTION": AMBER_LT, "DENIES": ABOVE}
DISCLOSURE_HATCH = {}
