/**
 * Start the Next.js dev server and the Python function server together.
 *
 * Two processes because that is what production is: TypeScript routes and
 * Python functions are separate runtimes on Vercel, and developing against a
 * single merged process would hide exactly the boundary bugs that matter.
 */

import { spawn } from "node:child_process";

const children = [];

function start(name, command, args) {
  const child = spawn(command, args, {
    stdio: "inherit",
    shell: process.platform === "win32",
  });
  child.on("exit", (code) => {
    if (code !== 0 && code !== null) {
      console.error(`\n${name} exited with code ${code}`);
      shutdown(code);
    }
  });
  children.push(child);
  return child;
}

function shutdown(code = 0) {
  for (const child of children) {
    if (!child.killed) child.kill();
  }
  process.exit(code);
}

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));

const python = process.env.PYTHON || (process.platform === "win32" ? "py" : "python3");
start("python", python, ["scripts/dev_python.py", "--port", "8000"]);
start("next", "npx", ["next", "dev"]);
