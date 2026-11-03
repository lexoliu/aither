use std::path::Path;

/// Stands in for `uv python install <request>`: installs a copy of this
/// program as the interpreter, with the entries the real `uv` leaves around it.
#[cfg(unix)]
fn install_python(request: &std::ffi::OsStr) {
    use std::os::unix::fs::PermissionsExt;
    let install_dir = std::path::PathBuf::from(std::env::var_os("UV_PYTHON_INSTALL_DIR").unwrap());
    let tree = install_dir.join(request);
    let bin = tree.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    std::fs::create_dir_all(install_dir.join(".temp")).unwrap();
    let interpreter = bin.join("python3.13");
    std::fs::copy(std::env::current_exe().unwrap(), &interpreter).unwrap();
    std::fs::set_permissions(&interpreter, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::os::unix::fs::symlink("python3.13", bin.join("python3")).unwrap();
    std::os::unix::fs::symlink(&tree, install_dir.join("cpython-3.13-minor")).unwrap();
    std::fs::write(install_dir.join(".lock"), b"").unwrap();
}

fn main() {
    let mut args = std::env::args_os();
    let program = args.next().unwrap();
    let name = Path::new(&program).file_name().unwrap().to_str().unwrap();
    // The managed shell reports its banner; any other invocation is the
    // host's bash, so scripts launched through it still run.
    #[cfg(unix)]
    if name == "bash" {
        use std::os::unix::process::CommandExt;
        let args = args.collect::<Vec<_>>();
        if args.first().is_some_and(|argument| argument == "--version") {
            println!("GNU bash, version 1.0.0(1)-release (fixture)");
            return;
        }
        let error = std::process::Command::new("/bin/bash").args(args).exec();
        panic!("cannot run the host's bash: {error}");
    }
    // The installed interpreter is a copy of this program.
    if name.starts_with("python") {
        println!("Python {}", env!("BASELINE_PYTHON_VERSION"));
        return;
    }
    // Invoked through a coreutils alias, behave as the host's command of that
    // name, as the multicall binary does.
    #[cfg(unix)]
    if !env!("BASELINE_COMMANDS").lines().any(|command| command == name) {
        use std::os::unix::process::CommandExt;
        let error = std::process::Command::new(name)
            .args(args)
            .env("PATH", "/usr/bin:/bin")
            .exec();
        panic!("cannot run the host's `{name}`: {error}");
    }
    let args = args.collect::<Vec<_>>();
    #[cfg(unix)]
    if name == "uv" && args.first().is_some_and(|argument| argument == "python") {
        install_python(args.last().unwrap());
        return;
    }
    if args.first().is_some_and(|argument| argument == "--list") {
        // The commands a multicall tool provides, one per line.
        println!("{}", env!("BASELINE_ALIASES"));
    } else {
        println!("baseline-tool 1.0.0");
    }
}
