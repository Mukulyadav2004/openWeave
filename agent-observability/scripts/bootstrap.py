"""Create a project and an API key, and print the credentials once.

    python scripts/bootstrap.py "my-app"

The secret is shown exactly once because only its hash is stored — that is the
point of the design, and it is worth keeping rather than "fixing" the first time
you lose a key. Re-run to mint another.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))

from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from auth import display_suffix, generate_key_pair, hash_secret  # noqa: E402
from models import ApiKey, Project  # noqa: E402

DATABASE_URL = (
    os.getenv("DATABASE_URL", "postgresql://openweave:openweave@localhost:5432/openweave")
    .replace("+asyncpg", "")
)


def main() -> None:
    project_name = sys.argv[1] if len(sys.argv) > 1 else "default"
    engine = create_engine(DATABASE_URL)

    with Session(engine) as session:
        project = session.scalar(select(Project).where(Project.name == project_name))
        if project is None:
            project = Project(name=project_name)
            session.add(project)
            session.flush()
            print(f"created project {project_name!r}")
        else:
            print(f"using existing project {project_name!r}")

        public_key, secret_key = generate_key_pair()
        session.add(
            ApiKey(
                project_id=project.id,
                public_key=public_key,
                hashed_secret_key=hash_secret(secret_key),
                display_secret_key=display_suffix(secret_key),
                note="created by bootstrap.py",
            )
        )
        session.commit()
        project_id = project.id

    print(f"""
  project_id  {project_id}
  public key  {public_key}
  secret key  {secret_key}   <-- shown once, only its hash is stored

  export OPENWEAVE_PUBLIC_KEY={public_key}
  export OPENWEAVE_SECRET_KEY={secret_key}
""")


if __name__ == "__main__":
    main()
