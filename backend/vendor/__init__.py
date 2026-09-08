"""Third-party parsers, vendored unmodified.

``csi_parse.py`` and ``csitool.py`` are an independent MediaTek MT7921 parser
supplied alongside this project. ``csi_dump_parsing.py`` is the LG on-board
presence detector -- a live tool that arms the radio, reads
/proc/net/wlan/csi_data and prints "+"/"-"; ``scripts/lg_detect_replay.py``
drives it over a recorded capture instead. All three are kept BYTE FOR BYTE as
received:
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

The detector additionally requires NUMPY 1.x, which is why ``.venv-board``
exists. Its TLV length arithmetic shifts a ``uint8`` left by 8: NumPy 1.x
promotes to int and yields 512 for the CSI tags, NumPy 2.x (NEP 50) keeps it
``uint8`` and yields 0, whereupon the walk desynchronises at the first CSI
field and produces frames with zeroed imaginary parts rather than an error.
The board runs 1.26.4. Create the venv with::

    uv venv --python 3.12 .venv-board
    VIRTUAL_ENV=.venv-board uv pip install numpy==1.26.4
"""
