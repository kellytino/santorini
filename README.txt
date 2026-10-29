Santorini Phase 1 - bot code and experiment runner

Requirements
- Python 3.10 or newer (3.11 recommended)
- No third-party packages required by the bots or experiment runner
- pytest is optional and required only to run the included tests

Directory layout
- src/santorini_engine.py, src/santorini_harness.py, src/santorini_types.py: copies of the instructor-provided files; do not modify these.
- src/random_bot.py: uniformly random legal action baseline.
- src/search_bot.py: minimax with alpha-beta pruning, iterative deepening, and a hand-coded evaluator.
- run_experiment.py: reproducible match runner writing CSV, JSON move logs, and configuration metadata.
- tests/test_bots.py: basic legality and immediate-win tests.

Install / run (Windows PowerShell or macOS/Linux)
1. Install Python 3.11 from https://www.python.org/downloads/ if needed.
2. Unzip the submission archive.
3. (Optional tests) python -m pip install pytest
4. From the santorini_phase1 folder, run:
   python -m pytest tests
5. Run one local match using the provided official CLI:
   python src/run_santorini.py --bot1 src/search_bot.py --bot2 src/random_bot.py --seed 20261028 --time-limit 1.5
6. Run the 50-game experiment:
   python run_experiment.py --games 50 --seed 20261028 --time-limit 1.5 --out results

The experiment alternates which bot is Player A and uses seeds 20261028 through 20261077 by default. It writes results/match_results.csv, results/move_logs.json, and results/config.json. The script prints a one-line summary per game. Search depth is iterative: the bot attempts depths 1, 2, and 3, and returns the best action from the deepest completed iteration before its time budget.

Important
- The official engine/harness files are included unchanged from the provided repository.
- Do not claim experiment results until you have run the script and inspected the output.
- The report should state the actual number of completed games and discuss any forfeits/timeouts.
