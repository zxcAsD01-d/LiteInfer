use rand::RngCore;
use rusqlite::{Connection, params};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

pub fn timestamp() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

pub fn hash(secret: &str) -> String {
    format!("{:x}", Sha256::digest(secret.as_bytes()))
}

#[derive(Clone)]
pub struct Store {
    path: PathBuf,
}

#[derive(Debug, Serialize)]
pub struct KeyInfo {
    pub id: String,
    pub name: String,
    pub created: f64,
    pub revoked: bool,
}

#[derive(Debug, Serialize)]
pub struct CreatedKey {
    pub id: String,
    pub key: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RequestRecord {
    pub id: String,
    pub key_id: String,
    pub model: String,
    pub protocol: String,
    pub status: u16,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub duration_ms: u64,
    pub created: f64,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self, rusqlite::Error> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
        }
        let store = Self { path: path.into() };
        let db = store.connect()?;
        db.execute_batch("PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS keys(id TEXT PRIMARY KEY,name TEXT NOT NULL,hash TEXT UNIQUE NOT NULL,created REAL NOT NULL,revoked INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY,key_id TEXT,model TEXT,protocol TEXT,status INTEGER,prompt_tokens INTEGER,completion_tokens INTEGER,duration_ms INTEGER,created REAL);
            CREATE INDEX IF NOT EXISTS requests_created ON requests(created);")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
        }
        Ok(store)
    }

    fn connect(&self) -> Result<Connection, rusqlite::Error> {
        let db = Connection::open(&self.path)?;
        db.busy_timeout(std::time::Duration::from_secs(5))?;
        Ok(db)
    }

    pub fn create_key(&self, name: &str) -> Result<CreatedKey, rusqlite::Error> {
        let mut bytes = [0_u8; 32];
        rand::rng().fill_bytes(&mut bytes);
        let secret = format!(
            "li_{}",
            bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()
        );
        let id = uuid::Uuid::new_v4().to_string();
        self.connect()?.execute(
            "INSERT INTO keys(id,name,hash,created) VALUES(?1,?2,?3,?4)",
            params![id, name, hash(&secret), timestamp()],
        )?;
        Ok(CreatedKey { id, key: secret })
    }

    pub fn revoke_key(&self, id: &str) -> Result<bool, rusqlite::Error> {
        Ok(self
            .connect()?
            .execute("UPDATE keys SET revoked=1 WHERE id=?1 AND revoked=0", [id])?
            > 0)
    }

    pub fn keys(&self) -> Result<Vec<KeyInfo>, rusqlite::Error> {
        let db = self.connect()?;
        let mut statement =
            db.prepare("SELECT id,name,created,revoked FROM keys ORDER BY created DESC")?;
        statement
            .query_map([], |r| {
                Ok(KeyInfo {
                    id: r.get(0)?,
                    name: r.get(1)?,
                    created: r.get(2)?,
                    revoked: r.get(3)?,
                })
            })?
            .collect()
    }

    pub fn active_keys(&self) -> Result<HashMap<String, String>, rusqlite::Error> {
        let db = self.connect()?;
        let mut statement = db.prepare("SELECT hash,id FROM keys WHERE revoked=0")?;
        statement
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect()
    }

    pub(crate) fn writer_connection(&self) -> Result<Connection, rusqlite::Error> {
        self.connect()
    }

    pub(crate) fn record_batch(
        db: &mut Connection,
        records: &[RequestRecord],
    ) -> Result<(), rusqlite::Error> {
        let transaction = db.transaction()?;
        let mut statement = transaction
            .prepare_cached("INSERT INTO requests VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9)")?;
        for record in records {
            statement.execute(params![
                record.id,
                record.key_id,
                record.model,
                record.protocol,
                record.status,
                record.prompt_tokens,
                record.completion_tokens,
                record.duration_ms,
                record.created
            ])?;
        }
        drop(statement);
        transaction.commit()
    }

    pub fn recent(&self) -> Result<Vec<RequestRecord>, rusqlite::Error> {
        let db = self.connect()?;
        let mut statement = db.prepare("SELECT * FROM requests ORDER BY created DESC LIMIT 100")?;
        statement
            .query_map([], |r| {
                Ok(RequestRecord {
                    id: r.get(0)?,
                    key_id: r.get(1)?,
                    model: r.get(2)?,
                    protocol: r.get(3)?,
                    status: r.get(4)?,
                    prompt_tokens: r.get(5)?,
                    completion_tokens: r.get(6)?,
                    duration_ms: r.get(7)?,
                    created: r.get(8)?,
                })
            })?
            .collect()
    }

    pub fn prune(&self) -> Result<(), rusqlite::Error> {
        self.connect()?.execute("DELETE FROM requests WHERE id IN (SELECT id FROM requests ORDER BY created DESC LIMIT -1 OFFSET 10000)", [])?;
        Ok(())
    }
}
