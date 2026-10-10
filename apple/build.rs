//! Build script for `aither-apple`.
//!
//! Compiles the Swift C-ABI bridge in `swift/` into a static archive when the
//! target is macOS or iOS. On every other target (including Linux docs.rs
//! builds) the crate compiles without the bridge: the public API is still
//! documented and `Availability` reports `UnsupportedOs` at runtime. No stub
//! `LanguageModel` implementation exists off Apple platforms.
//!
//! `DEVELOPER_DIR` and `SDKROOT` are honored through `xcrun`; a missing Xcode
//! toolchain or SDK fails the build with a diagnostic instead of degrading.
//! Xcode 27 is required for the public context-size query, which is
//! back-deployed to OS 26. Newer runtime features remain availability-gated.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Minimum deployment version: Foundation Models first shipped in 26.0.
const DEPLOYMENT_TARGET: &str = "26.0";

fn main() {
    println!("cargo::rustc-check-cfg=cfg(aither_apple_native)");
    println!("cargo::rustc-check-cfg=cfg(aither_sdk27)");
    println!("cargo::rustc-check-cfg=cfg(aither_scripted)");
    println!("cargo::rerun-if-changed=swift");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-env-changed=SDKROOT");
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    let target = env::var("TARGET").expect("TARGET is set by cargo");
    let Some((sdk_name, swift_target)) = apple_target(&target) else {
        // Not a supported Apple platform: the crate still compiles, and the
        // runtime API reports `UnavailableReason::UnsupportedOs`.
        return;
    };

    if env::var_os("DOCS_RS").is_some() {
        // docs.rs builds the crate for its own documentation pipeline; the
        // public API must document without a native toolchain present.
        return;
    }

    check_deployment(sdk_name);

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));

    let swiftc = xcrun(&["-f", "swiftc"]).unwrap_or_else(|e| {
        panic!(
            "aither-apple: cannot locate `swiftc` for target {target}: {e}. \
             Install Xcode 27 or newer (Foundation Models SDK), or set \
             DEVELOPER_DIR to a suitable toolchain."
        )
    });
    let sdk_path = sdk_path(sdk_name).unwrap_or_else(|e| {
        panic!(
            "aither-apple: cannot locate the {sdk_name} SDK for target \
             {target}: {e}. Check DEVELOPER_DIR/SDKROOT or install the SDK \
             platform in Xcode."
        )
    });

    let mut defines = sdk_defines(&sdk_path, &target);
    let has_sdk27 = defines.iter().any(|d| d == "AITHER_SDK_27");
    assert!(
        has_sdk27,
        "aither-apple requires Xcode 27 SDK for public contextSize metadata; deployment remains OS 26"
    );
    // Let tests know whether the scripted-model API surface was compiled in.
    if has_sdk27 {
        println!("cargo:rustc-cfg=aither_sdk27");
    }

    // The scripted executor requires explicit test-feature opt-in.
    if env::var_os("CARGO_FEATURE_TEST").is_some() {
        defines.push("AITHER_SCRIPTED".to_string());
        println!("cargo:rustc-cfg=aither_scripted");
    }

    let sources = swift_sources();
    let archive = out_dir.join("libaither_apple_bridge.a");

    let mut command = Command::new(&swiftc);
    command
        .arg("-emit-library")
        .arg("-static")
        .arg("-parse-as-library")
        .arg("-swift-version")
        .arg("6")
        .arg("-O")
        .arg("-module-name")
        .arg("aither_apple_bridge")
        .arg("-target")
        .arg(&swift_target)
        .arg("-sdk")
        .arg(&sdk_path);
    for define in defines {
        command.arg("-D").arg(define);
    }
    let status = command
        .args(&sources)
        .arg("-o")
        .arg(&archive)
        .status()
        .expect("aither-apple: failed to spawn swiftc");
    assert!(
        status.success(),
        "aither-apple: swiftc exited with {status} for target {target}"
    );

    println!("cargo:rustc-cfg=aither_apple_native");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=aither_apple_bridge");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=CoreGraphics");
    println!("cargo:rustc-link-lib=framework=ImageIO");
    println!("cargo:rustc-link-lib=framework=FoundationModels");
    println!("cargo:rustc-link-lib=framework=UniformTypeIdentifiers");

    // Swift runtime resolution happens through the SDK's `/usr/lib/swift`
    // stubs, whose install names are the OS-absolute paths present in the
    // dyld shared cache — so final executables need no rpath fixes. The
    // toolchain directory is intentionally absent: its `.tbd`s carry the
    // back-deployment `@rpath` install names, which would force every
    // consumer to add rpaths. A 26.0 minimum deployment also selects the
    // absolute install name through the SDK stubs' $ld$previous$ rules.
    let sdk_swift = Path::new(&sdk_path).join("usr/lib/swift");
    assert!(
        sdk_swift.exists(),
        "aither-apple: {sdk_path} has no usr/lib/swift stub directory"
    );
    println!("cargo:rustc-link-search=native={}", sdk_swift.display());
    println!("cargo:rustc-link-search=native=/usr/lib/swift");
    println!("cargo:rustc-link-lib=swiftCore");
    println!("cargo:rustc-link-lib=swift_Concurrency");
    // StringProcessing backs `Regex`, used by generation guides.
    println!("cargo:rustc-link-lib=swift_StringProcessing");
    println!("cargo:rustc-link-lib=swift_RegexParser");
}
fn check_deployment(sdk_name: &str) {
    let deployment_env = if sdk_name == "macosx" {
        "MACOSX_DEPLOYMENT_TARGET"
    } else {
        "IPHONEOS_DEPLOYMENT_TARGET"
    };
    println!("cargo::rerun-if-env-changed={deployment_env}");
    let deployment = env::var(deployment_env).unwrap_or_else(|_| {
        panic!("aither-apple: set {deployment_env}=26.0 (or newer) for every Cargo final target, including downstream applications")
    });
    let major = deployment
        .split('.')
        .next()
        .and_then(|v| v.parse::<u32>().ok())
        .expect("aither-apple: deployment target must be a version number");
    assert!(
        major >= 26,
        "aither-apple: {deployment_env} must be at least 26.0"
    );
}

/// Maps a Rust target triple to (SDK name, Swift target triple), or `None`
/// for platforms where the provider is unavailable.
///
/// Note `x86_64-apple-ios` *is* the Intel iOS simulator target; the
/// `-sim`-suffixed triple only exists for aarch64.
fn apple_target(target: &str) -> Option<(&'static str, String)> {
    let min = DEPLOYMENT_TARGET;
    match target {
        "aarch64-apple-darwin" => Some(("macosx", format!("arm64-apple-macos{min}"))),
        "x86_64-apple-darwin" => Some(("macosx", format!("x86_64-apple-macos{min}"))),
        "aarch64-apple-ios" => Some(("iphoneos", format!("arm64-apple-ios{min}"))),
        "x86_64-apple-ios" => Some((
            "iphonesimulator",
            format!("x86_64-apple-ios{min}-simulator"),
        )),
        "aarch64-apple-ios-sim" => {
            Some(("iphonesimulator", format!("arm64-apple-ios{min}-simulator")))
        }
        _ => None,
    }
}

fn swift_sources() -> Vec<PathBuf> {
    let dir = Path::new("swift");
    let mut sources: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(dir).expect("aither-apple: swift/ directory missing") {
        let path = entry.expect("readable swift/ entry").path();
        if path.extension().is_some_and(|ext| ext == "swift") {
            sources.push(path);
        }
    }
    sources.sort();
    sources
}

/// Probes the SDK's Foundation Models interface for post-26.0 API and emits
/// the matching `-D` flags, so the same sources compile against SDK 26
/// (framework baseline), SDK 26.4 (`DynamicGenerationSchema.null`) and
/// SDK 27 (`LanguageModel`/`LanguageModelExecutor`, `ToolCallingMode`,
/// `LanguageModelCapabilities`, transcript attachments, usage).
///
/// A missing or unreadable interface is a broken SDK, not a baseline — the
/// bridge cannot compile without the framework module anyway, so fail here
/// with a clear diagnostic.
fn sdk_defines(sdk_path: &str, target: &str) -> Vec<String> {
    let interface = find_swiftinterface(sdk_path, target).unwrap_or_else(|| {
        panic!(
            "aither-apple: no FoundationModels swiftinterface for {target} in \
             SDK {sdk_path}. Foundation Models metadata requires Xcode 27+."
        )
    });
    let text = std::fs::read_to_string(&interface)
        .unwrap_or_else(|e| panic!("aither-apple: cannot read {}: {e}", interface.display()));
    let mut defines = Vec::new();
    if text.contains("LanguageModelExecutorGenerationChannel") {
        defines.push("AITHER_SDK_27".to_string());
    }
    if text.contains("representNilExplicitlyInGeneratedContent") {
        defines.push("AITHER_SDK_26_4".to_string());
    }
    defines
}

/// Locates the `FoundationModels` swiftinterface matching the build target's
/// architecture inside the given SDK.
fn find_swiftinterface(sdk_path: &str, target: &str) -> Option<PathBuf> {
    let (arch, platform) = match target {
        "aarch64-apple-darwin" => ("arm64e", "apple-macos"),
        "x86_64-apple-darwin" => ("x86_64", "apple-macos"),
        "aarch64-apple-ios" => ("arm64e", "apple-ios"),
        "x86_64-apple-ios" | "aarch64-apple-ios-sim" => {
            // Simulator slices live under the "-simulator" swiftinterface.
            let arch = if target.starts_with("x86_64") {
                "x86_64"
            } else {
                "arm64"
            };
            return exact_interface(sdk_path, arch, "apple-ios-simulator");
        }
        _ => return None,
    };
    exact_interface(sdk_path, arch, platform)
}

fn exact_interface(sdk_path: &str, arch: &str, platform: &str) -> Option<PathBuf> {
    let candidate = Path::new(sdk_path)
        .join("System/Library/Frameworks/FoundationModels.framework/Modules/FoundationModels.swiftmodule")
        .join(format!("{arch}-{platform}.swiftinterface"));
    if candidate.exists() {
        return Some(candidate);
    }
    None
}

fn sdk_path(sdk_name: &str) -> Result<String, String> {
    if let Some(root) = env::var_os("SDKROOT") {
        let root = PathBuf::from(root);
        if root.is_absolute() {
            return Ok(root.to_string_lossy().into_owned());
        }
        return xcrun(&["--sdk", &root.to_string_lossy(), "--show-sdk-path"]);
    }
    xcrun(&["--sdk", sdk_name, "--show-sdk-path"])
}

fn xcrun(args: &[&str]) -> Result<String, String> {
    let output = Command::new("xcrun")
        .args(args)
        .output()
        .map_err(|e| format!("failed to run xcrun {args:?}: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "xcrun {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map(|s| s.trim().to_string())
        .map_err(|e| e.to_string())
}
