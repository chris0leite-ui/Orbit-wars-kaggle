# FLAG: knob tune in progress — resume after any container restart

If the background knob tuner is not running (check: `pgrep -f knob_tune`),
resume it with the SAME rng and the existing log (completed evals are
replayed, the search continues where it died):

    python scripts/knob_tune.py --referee submissions/_ns_veto_rf.py \
      --generations 5 --pop 6 --elite 2 --seeds-per-eval 3 \
      --max-steps 150 --workers 1 --rng 0 \
      --resume audit/tune/knob_tune_20260611T093744Z.jsonl

Prerequisites after a restart: rebuild the referee first —
    python scripts/bundle_producer_plus.py --variant veto_rf
    sed 's/PRODUCER_PLUS_/PPNSX_/g' submissions/producer_plus_veto_rf_on.py > submissions/_ns_veto_rf.py

The tuner now commits+pushes its log after every eval, so progress
survives restarts. Remove this flag when the tune completes (final BEST
line) and the winner has been queued for n>=32 confirmation.
