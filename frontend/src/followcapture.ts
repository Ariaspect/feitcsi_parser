/** Live following the capture controller through a batch.
 *
 * A batch writes one file per capture, under a stamp the controller reports as
 * each one starts, so a view fixed on one path goes quiet after the first.
 * While Live is on, this opens each capture as it starts. It acts when the
 * stamp changes rather than whenever the view differs from it, so a capture
 * picked by hand stays on screen until the next one starts.
 *
 * It also finds a capture again when its file moves. On the collection host
 * one being written is at live/<stamp>.bin and goes to <day>/<stamp>.bin when
 * it ends, so the path being polled turns into a 404.
 */

import type { CaptureFile } from "./api";

/** Shorter than this, a capture is not opened yet: about 1.5 s of frames. The
 *  backend picks a file's parser from its first bytes and keeps that choice,
 *  and the MIMO and MAC selections are reconciled once against the frames seen
 *  on opening, so a file that has barely begun would set both wrongly. */
export const MIN_OPEN_BYTES = 256 * 1024;
export const RETRY_MS = 2000;
/** A file that never turns up (recording without upload, a capture that was
 *  cancelled and removed) stops being searched for after about two minutes. */
export const MAX_TRIES = 60;

/** The stamp of a capture's path, `.../20261030_143000.bin` -> `20261030_143000`.
 *  Null for any other name: only a stamp is known to identify one capture. */
export function stampOfPath(path: string): string | null {
  const m = /(?:^|\/)(\d{8}_\d{6})\.[^./]+$/.exec(path);
  return m ? m[1] : null;
}

/** The copy of a capture to open. Mid-upload there can be two, the live copy
 *  and a partial one, and the longer is the one with more of it. */
export function findCapture(list: CaptureFile[], stamp: string): CaptureFile | undefined {
  let best: CaptureFile | undefined;
  for (const c of list) {
    if (stampOfPath(c.path) !== stamp || c.size_bytes < MIN_OPEN_BYTES) continue;
    if (!best || c.size_bytes > best.size_bytes) best = c;
  }
  return best;
}

export interface FollowInput {
  /** The Live button. */
  live: boolean;
  /** The capture the controller is writing; null between captures and when idle. */
  stamp: string | null;
  /** The capture on screen. */
  path: string;
  /** The last read of `path` failed. */
  failing: boolean;
}

export function createCaptureFollower(
  list: () => Promise<CaptureFile[]>,
  open: (capture: CaptureFile, all: CaptureFile[]) => void,
) {
  let followed: string | null = null;     // the controller stamp already acted on
  let searched: string | null = null;     // the failing path already looked for
  let search = 0;                         // the search in progress; a newer one orphans it
  let seeking = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let viewed = "";

  const cancel = () => {
    search += 1;
    seeking = false;
    clearTimeout(timer);
  };

  const look = async (id: number, stamp: string, tries: number) => {
    let all: CaptureFile[] | null = null;
    try {
      all = await list();
    } catch {
      // A failed listing counts as not found yet.
    }
    if (id !== search) return;
    const found = all && findCapture(all, stamp);
    if (all && found) {
      seeking = false;
      if (found.path !== viewed) open(found, all);
    } else if (tries < MAX_TRIES) {
      timer = setTimeout(() => void look(id, stamp, tries + 1), RETRY_MS);
    } else {
      seeking = false;
    }
  };

  const seek = (stamp: string) => {
    cancel();
    seeking = true;
    void look(search, stamp, 1);
  };

  const stop = () => {
    cancel();
    followed = null;
    searched = null;
  };

  return {
    update({ live, stamp, path, failing }: FollowInput) {
      viewed = path;
      if (!live) {
        stop();
        return;
      }
      if (stamp && stamp !== followed) {
        followed = stamp;
        seek(stamp);
        return;
      }
      if (!failing) {
        searched = null;
        return;
      }
      if (seeking || path === searched) return;
      searched = path;
      const own = stampOfPath(path);
      if (own) seek(own);
    },
    stop,
  };
}
