#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
#
# Combination Lock Analysis Suite (CLAS)
#
# An open-source utility for recording, visualizing, and analyzing mechanical
# combination lock measurements for educational and locksport purposes.
#
# Copyright (C) 2026 knowthebird
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 only.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# USE POLICY:
# This software is intended ONLY for:
#   - Educational use
#   - Locksport
#   - Locksmith training
#   - Locks that you own or have explicit permission to work on
#
# Misuse of this software may violate local, state, or federal law.
# The authors and contributors accept no liability for misuse.
#
# Module: clas_web.py
# Purpose: Web adapter stub (FastAPI endpoints around the same core engine).
#
# This is a reference adapter for future web UI work.
# Core logic MUST remain in clas_core.py.

"""
CLAS Web Adapter (Stub)

This module demonstrates a minimal web interface around the same CLAS core engine.

It is intentionally small and “boring”:
- It exposes endpoints that return the current prompt and accept actions.
- It keeps sessions in an in-memory dict in this stub version.

For production use you would replace the in-memory store with persistent storage
(files, SQLite/Postgres, Redis, etc.) while keeping the same core engine calls.

Navigation guide (search for these headers / sections):
  - Request/response models
  - Session store (in-memory stub)
  - Endpoints: create session, get prompt, apply action, undo
"""


from __future__ import annotations

from typing import Any, Dict
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import clas_core as core

app = FastAPI(title="CLAS Web Adapter (stub)")

# In-memory sessions (for demo). Replace with DB/file store later.
SESSIONS: Dict[str, Dict[str, Any]] = {}


class CreateSessionResponse(BaseModel):
    session_id: str
    session: Dict[str, Any]
    prompt: Dict[str, Any]


class ActionRequest(BaseModel):
    type: str  # "input" | "command"
    text: str | None = None
    name: str | None = None


class ActionResponse(BaseModel):
    session: Dict[str, Any]
    prompt: Dict[str, Any]


@app.post("/sessions", response_model=CreateSessionResponse)
def create_session() -> CreateSessionResponse:
    sid = str(uuid4())
    sess = core.new_session(session_name=f"web-{sid[:8]}")
    sess = core.normalize_session(sess)
    sess = core.rebuild(sess)
    SESSIONS[sid] = sess
    return CreateSessionResponse(session_id=sid, session=sess, prompt=core.get_prompt(sess))


@app.get("/sessions/{session_id}/prompt", response_model=Dict[str, Any])
def get_prompt(session_id: str) -> Dict[str, Any]:
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return core.get_prompt(sess)


@app.post("/sessions/{session_id}/action", response_model=ActionResponse)
def apply_action(session_id: str, req: ActionRequest) -> ActionResponse:
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    action = {"type": req.type}
    if req.type == "input":
        action["text"] = req.text or ""
    elif req.type == "command":
        action["name"] = (req.name or "").lower()
    else:
        raise HTTPException(status_code=400, detail="Invalid action type")

    sess2 = core.apply_action(sess, action)
    sess2 = core.normalize_session(sess2)
    SESSIONS[session_id] = sess2
    return ActionResponse(session=sess2, prompt=core.get_prompt(sess2))


@app.post("/sessions/{session_id}/undo", response_model=ActionResponse)
def undo(session_id: str) -> ActionResponse:
    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    sess2 = core.apply_action(sess, {"type":"command","name":"undo"})
    sess2 = core.normalize_session(sess2)
    SESSIONS[session_id] = sess2
    return ActionResponse(session=sess2, prompt=core.get_prompt(sess2))
