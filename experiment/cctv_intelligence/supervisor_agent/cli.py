"""Call a tool from the shell (for developers and for agent tests).

  python -m supervisor_agent.cli list_cameras
  python -m supervisor_agent.cli get_scene_log '{"camera":"cam07","start":"10:00","end":"10:10"}'
"""
import json
import sys

from . import db as D
from .tools import TOOLS, call


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in TOOLS:
        print("usage: python -m supervisor_agent.cli <tool> '<json args>'\ntools:", ", ".join(TOOLS))
        sys.exit(1)
    args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    res = call(D.connect(), sys.argv[1], args)
    print(res["text"])
    for p in res["images"]:
        print("[image]", p)


if __name__ == "__main__":
    main()
