"""Shared colours for every figure. One place to change them.

  blue    = above the threshold pays, and the donation bet itself
  amber   = below the threshold pays
  berry   = the instructed condition
  grey    = a single-series chart, or the no-bet baseline (baseline always dashed)

Blue does double duty on purpose: the bet is the subject of every experiment and it
mostly aims above the threshold, so the same hue reads as "the bet" where no side
split is shown and as "above pays" where one is. Amber therefore appears in only two
figures, and everything else is blue, berry or neutral.

Where a figure needs a second split it uses FILL — a filled mark against a hollow one
— and its legend says what the fill means there. Lightness is never used for this:
two steps of one hue are not reliably distinguishable at scatter-mark size, and light
steps of the saturated hues fail the chroma floor outright.

Colour encodes nothing when the x-axis already separates the categories. The ladder
puts its prompt families on the x-axis, so it is drawn in one neutral rather than
repeating that information in hue.

Every value was checked with a contrast and colour-blindness validator rather than by
eye. Blue, amber and berry pass together on lightness band, chroma floor, all-pairs
CVD separation (worst dE 14.9 under protanopia) and normal-vision floor. Candidates
that were tried and rejected: a green, which fell to dE 7.4 against blue under
tritanopia; a brighter rose, weaker under protanopia; and a burnt sienna, which had
the best numbers but read as amber's sibling and blurred the family/side distinction.
"""

# --- the two sides of the bet ---------------------------------------------
ABOVE_PAYS = "#3d4fa8"   # blue
BELOW_PAYS = "#c77a00"   # amber

# --- the two prompt families ----------------------------------------------
BET = ABOVE_PAYS         # the same blue
INSTRUCTED = "#9c3050"   # deep berry

# Light tints, for a second split INSIDE one figure: a lighter fill against a darker
# one, always with a legend beside it. Not for telling series apart across a plot —
# two steps of one hue are unreliable at small mark size, which is why the primary
# encodings above are separate hues.
BET_LIGHT = "#b5c2e9"
INSTRUCTED_LIGHT = "#e0a7b5"

# --- neutrals -------------------------------------------------------------
NEUTRAL = "#46536b"      # single-series charts, where colour encodes nothing
BASELINE = "#8a9199"     # no bet — drawn dashed, never colour alone
WHITE = "#ffffff"

SIDE = {"above": ABOVE_PAYS, "below": BELOW_PAYS}
FAMILY = {"bet": BET, "instructed": INSTRUCTED}

# Keys match the repo's condition names so this drops into existing code.
COLORS = {"baseline": BASELINE, "below_good": BELOW_PAYS, "above_good": ABOVE_PAYS}

# --- disclosure: an ordered scale, so a single-hue ramp --------------------
# The four labels run from open about the incentive to denying it. That is an order,
# not four unrelated categories, so it takes a sequential ramp rather than categorical
# hues. Blue because every trace being labelled is a bet trace.
# "Never raises it" sits OFF the ramp, in grey. It is not a further step along the
# open-to-closed scale — a trace that never mentions the incentive is silent rather
# than more closed than one that mentions it — and it is both the rarest label and
# the least reliable one. Marking it as a different kind of thing is honest, and it
# also keeps the three real steps far enough apart to tell apart.
DISCLOSURE = {"ADMITS": "#a2b1d9", "MENTIONS": "#4a5fae",
              "NO_MENTION": "#c9ccd1", "DENIES": "#2b3a7d"}
DISCLOSURE_RAMP = [DISCLOSURE[k] for k in ("ADMITS", "MENTIONS",
                                           "NO_MENTION", "DENIES")]
FAMILY_LIGHT = {"bet": BET_LIGHT, "instructed": INSTRUCTED_LIGHT}
DISCLOSURE_HATCH = {}
