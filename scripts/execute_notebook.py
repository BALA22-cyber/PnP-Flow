#!/usr/bin/env python3
"""Execute a notebook with the current environment's Jupyter kernel.

This repository's torchlearn environment includes ipykernel/jupyter_client but
not nbconvert/nbclient. This small runner provides the execution subset needed
for reproducible local notebook validation and writes cell outputs in place.
"""
from __future__ import annotations

import argparse
import queue
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager


def output_from_message(message: dict) -> dict | None:
    message_type = message["header"]["msg_type"]
    content = message["content"]
    if message_type == "stream":
        return nbformat.v4.new_output(
            "stream",
            name=content["name"],
            text=content["text"],
        )
    if message_type in {"display_data", "update_display_data"}:
        return nbformat.v4.new_output(
            "display_data",
            data=content["data"],
            metadata=content.get("metadata", {}),
        )
    if message_type == "execute_result":
        return nbformat.v4.new_output(
            "execute_result",
            execution_count=content["execution_count"],
            data=content["data"],
            metadata=content.get("metadata", {}),
        )
    if message_type == "error":
        return nbformat.v4.new_output(
            "error",
            ename=content["ename"],
            evalue=content["evalue"],
            traceback=content["traceback"],
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    notebook = nbformat.read(notebook_path, as_version=4)
    manager = KernelManager(kernel_name=notebook.metadata.kernelspec.name)
    manager.start_kernel(cwd=str(args.workdir.resolve()))
    client = manager.client()
    client.start_channels()

    try:
        client.wait_for_ready(timeout=120)
        execution_count = 0
        code_cells = sum(cell.cell_type == "code" for cell in notebook.cells)
        code_index = 0

        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            code_index += 1
            execution_count += 1
            cell.outputs = []
            cell.execution_count = execution_count
            print(f"Executing code cell {code_index}/{code_cells}", flush=True)
            message_id = client.execute(cell.source, store_history=True)
            failed = False

            while True:
                try:
                    message = client.get_iopub_msg(timeout=30)
                except queue.Empty:
                    print(f"  still running cell {code_index}...", flush=True)
                    continue

                if message.get("parent_header", {}).get("msg_id") != message_id:
                    continue
                message_type = message["header"]["msg_type"]
                if message_type == "status" and message["content"]["execution_state"] == "idle":
                    break
                if message_type == "clear_output":
                    cell.outputs = []
                    continue

                output = output_from_message(message)
                if output is not None:
                    cell.outputs.append(output)
                    if output.output_type == "stream":
                        print(output.text, end="", flush=True)
                    elif output.output_type == "error":
                        failed = True
                        print("\n".join(output.traceback), file=sys.stderr, flush=True)

            nbformat.write(notebook, notebook_path)
            if failed:
                raise RuntimeError(f"Notebook execution failed in code cell {code_index}")

        print(f"Executed notebook saved in place: {notebook_path}", flush=True)
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)


if __name__ == "__main__":
    main()
