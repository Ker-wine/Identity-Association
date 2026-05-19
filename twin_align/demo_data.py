"""文件内容：本文件包含最小样例数据生成逻辑。
主要职责：负责生成 demo users.csv/posts.csv。
前置文件：无。
后置文件：cli.py。
"""

from pathlib import Path
from typing import Any

import pandas as pd


def make_demo_data(args: Any) -> None:
    out = Path(args.outDir)
    out.mkdir(parents=True, exist_ok=True)
    users = pd.DataFrame(
        [
            ["u001", "twitter", "tw001", "alice_tw"],
            ["u001", "instagram", "ig001", "alice_ig"],
            ["u002", "twitter", "tw002", "bob_goals"],
            ["u002", "instagram", "ig002", "bob_football"],
            ["u003", "twitter", "tw003", "claire_code"],
            ["u003", "instagram", "ig003", "claire_dev"],
            ["u004", "twitter", "tw004", "david_food"],
            ["u004", "instagram", "ig004", "david_kitchen"],
        ],
        columns=["entityId", "platformId", "userId", "username"],
    )
    posts = pd.DataFrame(
        [
            ["twitter", "tw001", "p001", "coffee travel paris sunrise", "2017-01-01 09:00:00", "", ""],
            ["twitter", "tw001", "p002", "booking flight and reading cafe notes", "2017-01-02 09:30:00", "", ""],
            ["instagram", "ig001", "p003", "coffee time in paris morning trip", "2017-01-01 10:00:00", "", ""],
            ["instagram", "ig001", "p004", "sunrise cafe and travel diary", "2017-01-02 10:20:00", "", ""],
            ["twitter", "tw002", "p005", "football match tonight striker goal", "2017-01-01 21:00:00", "", ""],
            ["twitter", "tw002", "p006", "training day and league table", "2017-01-03 20:40:00", "", ""],
            ["instagram", "ig002", "p007", "great football day goal celebration", "2017-01-01 22:00:00", "", ""],
            ["instagram", "ig002", "p008", "league match and team training", "2017-01-03 21:10:00", "", ""],
            ["twitter", "tw003", "p009", "python api model training notebook", "2017-01-01 01:00:00", "", ""],
            ["twitter", "tw003", "p010", "debugging sklearn pipeline features", "2017-01-02 02:00:00", "", ""],
            ["instagram", "ig003", "p011", "coding notebook and api experiment", "2017-01-01 01:20:00", "", ""],
            ["instagram", "ig003", "p012", "python model features debug night", "2017-01-02 02:10:00", "", ""],
            ["twitter", "tw004", "p013", "dinner recipe pasta basil kitchen", "2017-01-01 18:00:00", "", ""],
            ["twitter", "tw004", "p014", "market vegetables and soup recipe", "2017-01-02 17:40:00", "", ""],
            ["instagram", "ig004", "p015", "kitchen pasta recipe with basil", "2017-01-01 18:20:00", "", ""],
            ["instagram", "ig004", "p016", "vegetable soup from local market", "2017-01-02 18:00:00", "", ""],
        ],
        columns=["platformId", "userId", "postId", "text", "timestamp", "imagePath", "imageEmbedding"],
    )
    users.to_csv(out / "users.csv", index=False)
    posts.to_csv(out / "posts.csv", index=False)
    print(f"Wrote demo data to {out}")
