use std::path::Path;

fn main() {
    let mut args = std::env::args_os();
    let program = args.next().unwrap();
    let name = Path::new(&program).file_name().unwrap().to_str().unwrap();
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
    if args.next().is_some_and(|argument| argument == "--list") {
        // The commands a multicall tool provides, one per line.
        println!("{}", env!("BASELINE_ALIASES"));
    } else {
        println!("baseline-tool 1.0.0");
    }
}
