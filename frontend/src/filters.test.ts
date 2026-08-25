import { describe, expect, it } from "vitest";
import { DEFAULT_MIMO, pickMimo } from "./filters";

describe("pickMimo", () => {
  it("defaults to 2x1 when the capture offers it", () => {
    expect(pickMimo("all", ["1x1", "2x1"], false)).toBe(DEFAULT_MIMO);
    expect(pickMimo("all", ["2x1", "2x2"], false)).toBe(DEFAULT_MIMO);
  });

  it("falls back to all when the capture has no 2x1", () => {
    expect(pickMimo("all", ["1x1", "2x2"], false)).toBe("all");
    expect(pickMimo("all", [], false)).toBe("all");
  });

  it("does not overrule a choice the user made", () => {
    expect(pickMimo("all", ["1x1", "2x1"], true)).toBe("all");
    expect(pickMimo("2x2", ["2x1", "2x2"], true)).toBe("2x2");
  });

  it("drops a user choice the new capture cannot satisfy", () => {
    // Keeping 2x2 here would filter every frame away and draw an empty plot.
    expect(pickMimo("2x2", ["1x1", "2x1"], true)).toBe("all");
  });
});
