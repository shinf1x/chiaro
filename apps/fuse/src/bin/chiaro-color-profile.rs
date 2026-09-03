use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use chiaro_fusion::{calibration::LriMessages, color_profile::FactoryColorDump};
use clap::Parser;
use serde::Serialize;

#[derive(Debug, Serialize)]
struct FileReport {
    path: String,
    raw: FactoryColorDump,
    analysis: Option<chiaro_fusion::color_profile::FactoryColorAnalysis>,
}

#[derive(Debug, Parser)]
#[command(
    name = "chiaro-color-profile",
    version,
    about = "Dump Light L16 factory colour calibration records as JSON"
)]
struct Cli {
    /// Capture or calibration LRI files to inspect.
    #[arg(required = true)]
    input: Vec<PathBuf>,

    /// JSON output. Standard output is used when omitted.
    #[arg(long, short)]
    output: Option<PathBuf>,

    /// Omit matrix fitting and held-out validation, leaving only raw records.
    #[arg(long)]
    raw_only: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut files = Vec::with_capacity(cli.input.len());
    for path in &cli.input {
        let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let messages =
            LriMessages::parse(&bytes).with_context(|| format!("parse {}", path.display()))?;
        let raw = FactoryColorDump::from_messages(&messages);
        let analysis = (!cli.raw_only).then(|| raw.analyze());
        files.push(FileReport {
            path: path.display().to_string(),
            raw,
            analysis,
        });
    }
    let json = serde_json::to_vec_pretty(&files)?;
    if let Some(path) = cli.output {
        fs::write(&path, json).with_context(|| format!("write {}", path.display()))?;
    } else {
        println!("{}", String::from_utf8(json)?);
    }
    Ok(())
}
