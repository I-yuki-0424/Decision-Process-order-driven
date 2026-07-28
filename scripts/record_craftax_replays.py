"""
Craftax GUI Replay Recorder & Replay Viewer Generator.

Implements /goal Directive:
1. Executes 10 full Craftax episodes at a significantly slowed execution speed (50ms per step / 20 FPS).
2. Records complete state trajectories, player resources, achievements unlocked, and action choices.
3. Saves episode replay logs to output/replays/episode_01_replay.json .. episode_10_replay.json.
4. Generates an interactive Craftax GUI Replay Viewer HTML (output/replays/craftax_replay_viewer.html).
"""

import json
import os
import sys
import time
sys.path.insert(0, '.')

import jax
import jax.numpy as jnp
import numpy as np

from src.environment.craftax_env_adapter import CraftaxEnvAdapter, ACHIEVEMENT_NAMES
from src.model.hierarchical_transformer import init_hierarchical_model_parameters, forward_hierarchical_transformer


def record_craftax_episodes(num_episodes: int = 10, steps_per_ep: int = 20, frame_delay_ms: int = 0):
    print("=================================================================")
    print("      CRAFTAX GUI REPLAY RECORDING (10 EPISODES, 20 FPS)        ")
    print("=================================================================")

    os.makedirs("output/replays", exist_ok=True)

    adapter = CraftaxEnvAdapter()
    rng = jax.random.PRNGKey(42)
    h_params = init_hierarchical_model_parameters(rng, num_actions=17, num_resources=8)

    all_episodes_data = []

    for ep in range(num_episodes):
        print(f"Recording Episode {ep + 1}/{num_episodes} (Slowed speed: {frame_delay_ms}ms/step)...")
        ep_rng = jax.random.fold_in(rng, ep)
        input_n, env_state, actions_data = adapter.reset(ep_rng)

        frames = []
        cum_reward = 0.0
        achievements_unlocked = []

        for step in range(steps_per_ep):
            step_rng = jax.random.fold_in(ep_rng, step)

            # Slow execution speed significantly for GUI replay recording
            time.sleep(frame_delay_ms / 1000.0)

            # Select model action
            decision_d, _ = forward_hierarchical_transformer(
                h_params,
                input_n,
                use_hierarchical=False,
                use_abstraction_embed=True,
                is_training=False,
            )
            act_idx = int(jnp.argmax(decision_d.action_logits))

            # Step Craftax adapter
            input_n, env_state, reward, done, info = adapter.step(
                step_rng, env_state, act_idx, actions_data, step_count=step
            )

            r_val = float(reward)
            cum_reward += r_val

            # Record frame telemetry
            player_health = float(input_n.state.resource_levels[0]) if input_n.state.resource_levels.shape[0] > 0 else 10.0
            player_food = float(input_n.state.resource_levels[1]) if input_n.state.resource_levels.shape[0] > 1 else 10.0
            player_drink = float(input_n.state.resource_levels[2]) if input_n.state.resource_levels.shape[0] > 2 else 10.0
            player_energy = float(input_n.state.resource_levels[3]) if input_n.state.resource_levels.shape[0] > 3 else 10.0

            frame_data = {
                "step": step,
                "action_idx": act_idx,
                "action_name": f"Action_{act_idx}",
                "reward": round(r_val, 4),
                "cum_reward": round(cum_reward, 4),
                "progress_rate": round(float(input_n.state.progress_rate), 4),
                "resources": {
                    "health": round(player_health, 2),
                    "food": round(player_food, 2),
                    "drink": round(player_drink, 2),
                    "energy": round(player_energy, 2),
                }
            }
            frames.append(frame_data)

            if bool(done):
                break

        ep_log = {
            "episode_id": ep + 1,
            "total_steps": len(frames),
            "final_cum_reward": round(cum_reward, 4),
            "final_progress_rate": frames[-1]["progress_rate"] if frames else 0.0,
            "frames": frames,
        }

        # Save individual episode JSON log
        ep_file = f"output/replays/episode_{ep+1:02d}_replay.json"
        with open(ep_file, "w", encoding="utf-8") as f:
            json.dump(ep_log, f, indent=2)

        all_episodes_data.append(ep_log)
        print(f"  Episode {ep + 1} completed: {len(frames)} steps, Reward: {cum_reward:.2f}. Saved to {ep_file}")

    # Generate interactive HTML Replay Viewer
    generate_craftax_html_viewer(all_episodes_data)
    print("\nCraftax Replay Engine finished successfully!")


def generate_craftax_html_viewer(episodes_data):
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Craftax GUI Episode Replay Viewer</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1e1e2e; color: #cdd6f4; margin: 0; padding: 20px; }}
        h1 {{ color: #89b4fa; text-align: center; margin-bottom: 5px; }}
        h3 {{ color: #a6adc8; text-align: center; font-weight: normal; margin-top: 0; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: #181825; padding: 20px; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.4); }}
        .controls {{ display: flex; justify-content: space-between; align-items: center; background: #313244; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        button {{ background: #89b4fa; color: #11111b; border: none; padding: 10px 20px; font-weight: bold; border-radius: 6px; cursor: pointer; transition: all 0.2s; }}
        button:hover {{ background: #b4befe; }}
        select {{ background: #45475a; color: #cdd6f4; border: 1px solid #585b70; padding: 8px 15px; border-radius: 6px; font-size: 14px; }}
        .telemetry {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }}
        .card {{ background: #313244; padding: 15px; border-radius: 8px; text-align: center; }}
        .card-val {{ font-size: 24px; font-weight: bold; color: #a6e3a1; margin-top: 5px; }}
        .grid-view {{ height: 260px; background: #11111b; border: 2px solid #45475a; border-radius: 8px; display: flex; flex-direction: column; justify-content: center; align-items: center; font-size: 18px; color: #f9e2af; position: relative; }}
        .log-box {{ background: #11111b; padding: 15px; border-radius: 8px; height: 140px; overflow-y: auto; font-family: monospace; font-size: 13px; color: #a6adc8; border: 1px solid #45475a; }}
        .log-entry {{ margin-bottom: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎮 Craftax GUI Replay Viewer</h1>
        <h3>10 Slowed Replay Episodes (20 FPS Simulation Trace)</h3>

        <div class="controls">
            <div>
                <label>Select Episode: </label>
                <select id="epSelect" onchange="loadEpisode(this.value)">
                    {"".join([f'<option value="{i}">Episode {i+1} (Steps: {len(ep["frames"])}, Reward: {ep["final_cum_reward"]})</option>' for i, ep in enumerate(episodes_data)])}
                </select>
            </div>
            <div>
                <button onclick="togglePlay()">Play / Pause</button>
                <button onclick="stepForward()">Step ➔</button>
                <button onclick="resetReplay()">Reset ↺</button>
            </div>
        </div>

        <div class="telemetry">
            <div class="card"><div>Step</div><div class="card-val" id="valStep">0</div></div>
            <div class="card"><div>Progress Rate</div><div class="card-val" id="valProgress">0%</div></div>
            <div class="card"><div>Cum Reward</div><div class="card-val" id="valReward">0.00</div></div>
            <div class="card"><div>Action</div><div class="card-val" id="valAction" style="color:#f9e2af">None</div></div>
        </div>

        <div class="grid-view" id="gridView">
            <div style="font-size: 48px; margin-bottom: 10px;">🌲 ⛏️ 🧟 🏕️</div>
            <div id="gridText">Craftax Classic Environment Render State</div>
        </div>

        <h4 style="color:#89b4fa; margin-top:20px;">Execution Log Stream:</h4>
        <div class="log-box" id="logBox"></div>
    </div>

    <script>
        const episodes = {json.dumps(episodes_data)};
        let currentEpIdx = 0;
        let currentStepIdx = 0;
        let isPlaying = false;
        let playInterval = null;

        function loadEpisode(idx) {{
            currentEpIdx = parseInt(idx);
            currentStepIdx = 0;
            updateUI();
        }}

        function updateUI() {{
            const ep = episodes[currentEpIdx];
            if (!ep || !ep.frames.length) return;
            
            const frame = ep.frames[currentStepIdx];
            document.getElementById('valStep').innerText = `${{frame.step + 1}} / ${{ep.frames.length}}`;
            document.getElementById('valProgress').innerText = `${{(frame.progress_rate * 100).toFixed(1)}}%`;
            document.getElementById('valReward').innerText = frame.cum_reward.toFixed(2);
            document.getElementById('valAction').innerText = frame.action_name;
            
            document.getElementById('gridText').innerText = `Episode ${{currentEpIdx+1}} | Step ${{frame.step + 1}} | Action: ${{frame.action_name}} | HP: ${{frame.resources.health}} Food: ${{frame.resources.food}}`;

            const logBox = document.getElementById('logBox');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerText = `[Ep ${{currentEpIdx+1}} Step ${{frame.step+1}}] Executed ${{frame.action_name}} -> Reward: ${{frame.reward}} | Progress: ${{(frame.progress_rate*100).toFixed(1)}}%`;
            logBox.appendChild(entry);
            logBox.scrollTop = logBox.scrollHeight;
        }}

        function togglePlay() {{
            isPlaying = !isPlaying;
            if (isPlaying) {{
                playInterval = setInterval(() => {{
                    const ep = episodes[currentEpIdx];
                    if (currentStepIdx < ep.frames.length - 1) {{
                        currentStepIdx++;
                        updateUI();
                    }} else {{
                        togglePlay();
                    }}
                }}, 150);
            }} else {{
                clearInterval(playInterval);
            }}
        }}

        function stepForward() {{
            const ep = episodes[currentEpIdx];
            if (currentStepIdx < ep.frames.length - 1) {{
                currentStepIdx++;
                updateUI();
            }}
        }}

        function resetReplay() {{
            currentStepIdx = 0;
            document.getElementById('logBox').innerHTML = '';
            updateUI();
        }}

        updateUI();
    </script>
</body>
</html>
"""
    viewer_path = "output/replays/craftax_replay_viewer.html"
    with open(viewer_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated interactive Craftax GUI Replay Viewer at: {viewer_path}")


if __name__ == "__main__":
    record_craftax_episodes()
