# The platform a Rust target triple names, in the (os, arch) terms of the
# tool-distribution manifest: arch is the triple's first component, os is
# derived from the rest, not from a lookup table of whole triples.
def os_of: if endswith("-apple-darwin") then "macos"
           elif test("linux") then "linux"
           elif test("windows") then "windows"
           else error("cannot map target to an os") end;
def platform_of: {os: os_of, arch: split("-")[0]};
