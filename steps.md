Phase 1 — Data preparation

Assemble the full 1997–2023 five-minute archive from the DOI (you have one sample day; get the rest). Concatenate the per-day files into one continuous, time-ordered series, keeping only what you need: TimeStamp, Td (dry-bulb), Tw (wet-bulb), RH, P. The sample confirms these four are fully populated at 5-minute resolution.
Represent gaps as real gaps. Coverage is uneven — 2015 has only 161 days, 2023 stops on 4 May, some early years are short, and there were logging-system changeovers. Don't silently stitch across holes; a model trained across a hidden gap learns nonsense.
Set quality-control rules. The data already has range limits applied (out-of-range left blank). Decide how you handle blanks — skip any 5-minute row missing Td or Tw, and flag days with too few valid readings so they don't distort aggregates.

Phase 2 — Fix the target and the facility

Lock the forecast target: predict Td and Tw at exactly +3 hours (36 five-minute steps ahead). Start with the single +3h endpoint; extend to the full trajectory across the window only if time allows. Mode is a label computed from the predicted values, not a separately predicted thing.
Define the representative facility as fixed, defensible constants: IT load (e.g. 20 MW colocation), 100% utilisation, 7% electrical losses, cooling type (airside economiser + direct evaporative), and server-inlet setpoint (e.g. 24°C, inside ASHRAE's 18–27°C recommended band). All sourced, none invented.
Define the cooling-mode function: free cooling when predicted Td ≤ setpoint; evaporative when Td > setpoint but Tw is low enough; mechanical when Tw is too high. The forecast is learned; the thresholds are configurable assumptions from ASHRAE / Tetra Tech. Keep that boundary explicit — it's what keeps the tool interpretable.

Phase 3 — Feature engineering (where forecasting projects are won)

Build "knowable-now" features: current Td, Tw, RH, P; their lags (5, 15, 30, 60 min ago); short rolling means; and the recent trend, especially pressure tendency over the last hour.
Add cyclical time features: time-of-day and day-of-year encoded as sine/cosine, so the model distinguishes a July afternoon from a January night. High signal, low effort.
Guard against leakage obsessively. Every feature must be knowable at "now" — nothing from the 3-hour gap between now and the target may enter. This is the most common way forecasting projects quietly cheat.

Phase 4 — Model and validate honestly

Set baselines first: persistence (mode in 3h = mode now) and a time-of-day climatology. Your model must beat both to justify itself; failing to beat persistence is itself an honest, reportable result.
Train a gradient-boosting regressor (scikit-learn) predicting Td and Tw at +3h. Simple model, rich features — no neural nets, per the scope decision.
Validate time-ordered, never random: train on earlier years, test on the most recent available (e.g. 2020–2023). Report both the temperature error (°C) at +3h and the mode-classification accuracy.
Judge on the transitions, not the majority. Most of the UK year sits in free cooling, so raw accuracy looks high just by always guessing "free." Show how well the tool catches the rare free→evaporative→mechanical switches — those are the operationally valuable moments and the honest test.

Phase 5 — Turn it into a tool, and tell the story

Surface the operational output: current mode, predicted mode over the next 3 hours, when free cooling opens or closes, and a flag for any upcoming switch into evaporative/mechanical mode. The "heads-up before a switch" is the actual value proposition.
Quantify the consequence: translate a predicted switch into numbers using your facility spec — the jump in power (IT load × higher PUE) and water use when the facility leaves free cooling. This is what makes it a decision tool, not a thermometer.
Add the century context (Theme 3 tie-in): using the separate 1908–present daily record, show how the frequency of each cooling mode — especially free-cooling days per year — has shifted over 116 years as the climate warmed. Short-horizon tool in the foreground; long-run climate trend as the backdrop.git reset --soft HEAD~2
Frame limitations as rigour. State plainly: the facility is synthetic (no operator publishes real telemetry); PUE and thresholds come from published sources; the daily record's single 0900 reading is a proxy; and the 5-minute record ends in May 2023 — which is a non-issue because the model learns stable short-range atmospheric physics, not a time-specific pattern, so it applies to live data today. Judges reward this honesty.
Prepare the submission per the brief: public GitHub/GitLab repo with a one-page report naming the theme and datasets, plus a 10–15 minute presentation (a worked day where the tool correctly flags an incoming switch makes a strong demo).