import { spawnSync } from "node:child_process";
import process from "node:process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const scriptPath = path.join(scriptDir, "prepare_engine.py");

const commands = process.platform === "win32" ? ["python", "python3"] : ["python3", "python"];
let executed = false;

for (const command of commands) {
  const result = spawnSync(command, [scriptPath], { stdio: "inherit" });
  if (result.error && result.error.code === "ENOENT") {
    continue;
  }
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
  executed = true;
  break;
}

if (!executed) {
  console.error("Neither python3 nor python was found in PATH.");
  process.exit(1);
}
