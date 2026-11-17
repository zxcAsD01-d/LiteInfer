use clap::{Parser, Subcommand};
use liteinfer::{api, config::Config, state::AppState, store::Store};
use std::{path::PathBuf, process::ExitCode};

#[derive(Parser)]
#[command(name = "liteinfer", version, about = "macOS 本地轻量推理服务")]
struct Cli {
    #[arg(short, long, default_value = "config.toml", global = true)]
    config: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// 启动 Rust API 服务
    Serve,
    /// 校验配置；不会连接模型后端
    Check,
    /// 管理本地 API Key（运行中请使用控制台即时管理）
    Key {
        #[command(subcommand)]
        command: KeyCommand,
    },
}

#[derive(Subcommand)]
enum KeyCommand {
    Create {
        #[arg(long, default_value = "local")]
        name: String,
    },
    List,
    Revoke {
        id: String,
    },
}

#[tokio::main(worker_threads = 2)]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "liteinfer=info".into()),
        )
        .with_target(false)
        .init();
    match run(Cli::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("LiteInfer: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<(), String> {
    let config = if !cli.config.exists() && matches!(cli.command, Command::Serve) {
        let config = Config::default();
        let text = toml::to_string_pretty(&config).map_err(|_| "默认配置生成失败")?;
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&cli.config)
            .map_err(|_| "默认配置创建失败，文件可能已存在")?;
        file.write_all(text.as_bytes())
            .and_then(|_| file.sync_all())
            .map_err(|_| "默认配置写入失败")?;
        tracing::info!("已生成默认配置，可从网页控制台安装模型");
        config
    } else {
        Config::load(&cli.config)?
    };
    match cli.command {
        Command::Check => {
            println!(
                "配置有效：{} 个模型，并发 {}",
                config.models.len(),
                config.concurrency
            );
            Ok(())
        }
        Command::Key { command } => key_command(config, command),
        Command::Serve => serve(config).await,
    }
}

fn key_command(config: Config, command: KeyCommand) -> Result<(), String> {
    let store =
        Store::open(std::path::Path::new(&config.database)).map_err(|_| "数据库初始化失败")?;
    let value = match command {
        KeyCommand::Create { name } => {
            if name.trim().is_empty() || name.len() > 128 {
                return Err("Key 名称必须为 1..128 字节".into());
            }
            serde_json::to_value(store.create_key(&name).map_err(|_| "创建 Key 失败")?)
        }
        KeyCommand::List => serde_json::to_value(store.keys().map_err(|_| "列出 Key 失败")?),
        KeyCommand::Revoke { id } => {
            serde_json::to_value(store.revoke_key(&id).map_err(|_| "撤销 Key 失败")?)
        }
    }
    .map_err(|_| "Key 结果序列化失败")?;
    println!(
        "{}",
        serde_json::to_string_pretty(&value).map_err(|_| "序列化失败")?
    );
    Ok(())
}

async fn serve(config: Config) -> Result<(), String> {
    let address = config.listen;
    let admin_key = match std::env::var("LITEINFER_ADMIN_KEY") {
        Ok(value) => value,
        Err(_) => {
            use rand::RngCore;
            let mut bytes = [0u8; 32];
            rand::rng().fill_bytes(&mut bytes);
            let key: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
            eprintln!("当前进程管理密钥（请保存，在网页输入）：{key}");
            key
        }
    };
    let state = AppState::new(
        config,
        std::env::var("LITEINFER_API_KEY").ok(),
        Some(admin_key),
    )?;
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .map_err(|_| "监听端口失败，请检查端口占用")?;
    let restore_state = state.clone();
    tokio::spawn(async move {
        if let Err(error) = liteinfer::deployment::Deployment::restore(restore_state).await {
            tracing::error!("恢复 MLX 模型失败：{}", error.message);
        }
    });
    tracing::info!("LiteInfer 已启动，监听 {address}");
    let outcome = axum::serve(listener, api::router(state.clone()))
        .with_graceful_shutdown(shutdown())
        .await
        .map_err(|_| "HTTP 服务异常退出".to_string());
    state
        .deployment
        .close()
        .await
        .map_err(|_| "MLX 后端停止失败")?;
    outcome
}

async fn shutdown() {
    #[cfg(unix)]
    {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut terminate) => {
                tokio::select! {_=tokio::signal::ctrl_c()=>(),_=terminate.recv()=>()}
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
