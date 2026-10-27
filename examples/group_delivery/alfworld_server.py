# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-only ALFWorld environment service; install in its own Python environment.

Run: ALFWORLD_DATA=/data/alfworld python examples/group_delivery/alfworld_server.py
The default bind address is loopback. Use a single server worker because sessions are local.
"""

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


def make_environment(game_file, max_steps):
    """Construct the official text environment without privileged expert observations."""
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    env_id = textworld.gym.register_games(
        [str(game_file)],
        infos,
        batch_size=1,
        asynchronous=False,
        max_episode_steps=max_steps,
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
    )
    return textworld.gym.make(env_id)


class OpenSession(BaseModel):
    """Path relative to ALFWORLD_DATA, supplied by the prepared dataset."""

    game_file: str


class Action(BaseModel):
    """One environment action; model text never controls the reward."""

    action: str


def create_app(data_root, max_steps=50, environment_factory=make_environment):
    """Create isolated sessions; an injectable factory allows CPU lifecycle tests."""
    root = Path(data_root).resolve()
    sessions, registry_lock = {}, threading.RLock()

    @asynccontextmanager
    async def lifespan(app):
        yield
        for session in sessions.values():
            session["env"].close()

    app = FastAPI(lifespan=lifespan)

    def view(session):
        return {key: value for key, value in session.items() if key not in ("env", "lock")}

    def lookup(session_id):
        with registry_lock:
            if session_id not in sessions:
                raise HTTPException(404, "Unknown environment session")
            return sessions[session_id]

    @app.post("/sessions")
    def open_session(body: OpenSession):
        path = (root / body.game_file).resolve()
        if not path.is_relative_to(root) or path.name != "game.tw-pddl" or not path.is_file():
            raise HTTPException(400, "Expected a game.tw-pddl file inside ALFWORLD_DATA")
        with registry_lock:
            if len(sessions) >= 128:
                raise HTTPException(503, "Too many active environment sessions")
            env = environment_factory(path, max_steps)
            try:
                observations, infos = env.reset()
            except Exception:
                env.close()
                raise
            session_id = uuid4().hex
            sessions[session_id] = dict(
                env=env,
                lock=threading.RLock(),
                session_id=session_id,
                observation=observations[0],
                commands=list(infos["admissible_commands"][0]),
                steps=0,
                max_steps=max_steps,
                won=False,
                done=False,
                error=False,
            )
            return view(sessions[session_id])

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str):
        session = lookup(session_id)
        with session["lock"]:
            return view(session)

    @app.post("/sessions/{session_id}/step")
    def step_session(session_id: str, body: Action):
        session = lookup(session_id)
        with session["lock"]:
            if not session["done"]:
                try:
                    observations, _, dones, infos = session["env"].step([body.action])
                    session.update(
                        observation=observations[0],
                        commands=list(infos["admissible_commands"][0]),
                        won=bool(infos["won"][0]),
                        steps=session["steps"] + 1,
                    )
                    session["done"] = bool(dones[0]) or session["won"] or session["steps"] >= max_steps
                except Exception:
                    session.update(error=True, done=True)
                    raise
            return view(session)

    @app.delete("/sessions/{session_id}")
    def close_session(session_id: str):
        with registry_lock:
            session = sessions.pop(session_id, None)
        if session is not None:
            with session["lock"]:
                session["env"].close()
        return {"closed": True}

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        create_app(os.environ["ALFWORLD_DATA"], int(os.environ.get("ALFWORLD_MAX_STEPS", "50"))),
        host="127.0.0.1",
        port=8765,
    )
