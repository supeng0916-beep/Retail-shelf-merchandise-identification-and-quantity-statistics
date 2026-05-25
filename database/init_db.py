"""
初始化 SQLite 数据库 schema。
由 FastAPI lifespan 和 scripts/ 自动调用，幂等安全（CREATE TABLE IF NOT EXISTS）。
"""
import asyncio
from pathlib import Path

import aiosqlite

DB_PATH = Path(__file__).parent / "sku.db"
SKU_IMAGES_DIR = Path(__file__).parent / "sku_images"


def _db_uri(db_path: Path) -> str:
    # Use URI mode for more stable behavior on some Windows filesystems.
    return f"file:{db_path.as_posix()}?mode=rwc"


async def init_db(db_path: str | Path = DB_PATH) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    SKU_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(_db_uri(db_path), uri=True) as db:
        # Use in-memory journaling to avoid filesystem lock/journal issues.
        await db.execute("PRAGMA journal_mode=MEMORY")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.execute("PRAGMA temp_store=MEMORY")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS skus (
                sku_id   TEXT PRIMARY KEY,
                name     TEXT NOT NULL,
                barcode  TEXT,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sku_embeddings (
                faiss_idx INTEGER PRIMARY KEY,
                sku_id    TEXT NOT NULL,
                FOREIGN KEY (sku_id) REFERENCES skus(sku_id) ON DELETE CASCADE
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sku_embeddings_sku_id ON sku_embeddings(sku_id)"
        )
        await db.commit()


def init_db_sync(db_path: str | Path = DB_PATH) -> None:
    """同步版本，供非 async 脚本调用。"""
    import sqlite3
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    SKU_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_db_uri(db_path), uri=True)
    try:
        con.execute("PRAGMA journal_mode=MEMORY")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA temp_store=MEMORY")
        con.executescript("""
            CREATE TABLE IF NOT EXISTS skus (
                sku_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                barcode TEXT,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS sku_embeddings (
                faiss_idx INTEGER PRIMARY KEY,
                sku_id TEXT NOT NULL,
                FOREIGN KEY (sku_id) REFERENCES skus(sku_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_sku_embeddings_sku_id
                ON sku_embeddings(sku_id);
        """)
        con.commit()
    finally:
        con.close()


if __name__ == "__main__":
    asyncio.run(init_db())
    print(f"数据库初始化完成: {DB_PATH}")
