"""Third-party parsers, vendored unmodified.

``csi_parse.py`` and ``csitool.py`` are an independent MediaTek MT7921 parser
supplied alongside this project. They are kept here BYTE FOR BYTE as received:
the point of a second parser is that it is a second opinion, and editing it
would quietly turn it into an echo of our own. Fixes belong upstream, or in the
adapter that calls this.

Where the two disagree, both readings are worth having:

* Ours takes the ratio along ``tpi`` (lag-1 phase coherence 0.996 on
  20260904_192623); ``feature_conj`` takes it across ``rpi`` (0.885).
* Ours fftshifts to centre DC; this one leaves raw FFT order and labels its
  subcarrier index ``sub_k_provisional``.
* Ours interpolates pilots and DC and keeps 245 of 256 bins; this one takes
  bins by measured occupancy and keeps 234.
* This one filters to the dominant transmitter and restores absolute amplitude
  from RSSI, both of which ours gained after reviewing it.
"""
