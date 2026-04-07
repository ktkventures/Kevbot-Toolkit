"""Execution Types Router — list, inspect, and configure execution type modules."""

from fastapi import APIRouter, Depends, HTTPException, Body
from api.deps import get_current_user

router = APIRouter(prefix="/api/execution-types", tags=["execution-types"])


@router.get("")
def list_execution_types(user=Depends(get_current_user)):
    """List all registered execution type modules with user's enable/disable state."""
    from execution_types import list_modules

    # Load user's config (enabled state + parameter overrides)
    config = _load_config()

    variations = config.get('_variations', [])

    result = []
    for m in list_modules():
        d = m.to_dict()
        user_cfg = config.get(m.slug, {})
        d['enabled'] = user_cfg.get('enabled', m.slug in ('bar_close', 'level'))  # C and L enabled by default
        d['user_params'] = user_cfg.get('params', {})
        d['is_default'] = True
        d['variations'] = [v for v in variations if v.get('parent_slug') == m.slug]
        result.append(d)
    return result


@router.get("/config")
def get_config(user=Depends(get_current_user)):
    """Load user's execution type configuration."""
    return _load_config()


@router.put("/config")
def save_config(config: dict = Body(...), user=Depends(get_current_user)):
    """Save user's execution type configuration (enabled state + params)."""
    _save_config(config)
    return {"status": "saved"}


@router.get("/{slug}")
def get_execution_type(slug: str, user=Depends(get_current_user)):
    """Get a specific execution type module by slug."""
    from execution_types import list_modules
    config = _load_config()
    for m in list_modules():
        if m.slug == slug:
            d = m.to_dict()
            user_cfg = config.get(m.slug, {})
            d['enabled'] = user_cfg.get('enabled', m.slug in ('bar_close', 'level'))
            d['user_params'] = user_cfg.get('params', {})
            return d
    raise HTTPException(status_code=404, detail=f"Execution type '{slug}' not found")


@router.put("/{slug}/toggle")
def toggle_execution_type(slug: str, user=Depends(get_current_user)):
    """Toggle an execution type's enabled state."""
    config = _load_config()
    current = config.get(slug, {})
    current_enabled = current.get('enabled', slug in ('bar_close', 'level'))
    current['enabled'] = not current_enabled
    config[slug] = current
    _save_config(config)
    return {"slug": slug, "enabled": current['enabled']}


@router.put("/{slug}/params")
def update_params(slug: str, params: dict = Body(...), user=Depends(get_current_user)):
    """Update an execution type's parameter values."""
    config = _load_config()
    current = config.get(slug, {})
    current['params'] = params
    config[slug] = current
    _save_config(config)
    return {"slug": slug, "params": params}


# ---- Variations ----

@router.post("/{slug}/variations")
def create_variation(slug: str, body: dict = Body(...), user=Depends(get_current_user)):
    """Create a variation of an execution type with custom parameters."""
    from execution_types import list_modules
    parent = None
    for m in list_modules():
        if m.slug == slug:
            parent = m
            break
    if not parent:
        raise HTTPException(status_code=404, detail=f"Execution type '{slug}' not found")

    config = _load_config()
    variations = config.get('_variations', [])

    name = body.get('name', f'{parent.name} (Custom)')
    badge = body.get('badge', f'{parent.display_code}{len(variations) + 1}')
    params = body.get('params', {})
    var_id = f'{slug}_{len(variations) + 1}'

    variation = {
        'id': var_id,
        'parent_slug': slug,
        'name': name,
        'badge': badge,
        'params': params,
        'enabled': True,
    }
    variations.append(variation)
    config['_variations'] = variations
    _save_config(config)
    return variation


@router.delete("/variations/{var_id}")
def delete_variation(var_id: str, user=Depends(get_current_user)):
    """Delete a variation."""
    config = _load_config()
    variations = config.get('_variations', [])
    config['_variations'] = [v for v in variations if v.get('id') != var_id]
    _save_config(config)
    return {"status": "deleted", "id": var_id}


@router.get("/variations")
def list_variations(user=Depends(get_current_user)):
    """List all user-created variations."""
    config = _load_config()
    return config.get('_variations', [])


# ---- Storage helpers ----
# Store execution type config in user_settings under the key 'execution_types'

@router.get("/{slug}/code")
def get_execution_type_code(slug: str, user=Depends(get_current_user)):
    """Return the source code of an execution type module."""
    from execution_types import list_modules
    import inspect

    for m in list_modules():
        if m.slug == slug:
            source = inspect.getsource(m.__class__)
            return {
                'slug': slug,
                'name': m.name,
                'class_name': m.__class__.__name__,
                'source': source,
                'description': m.description,
                'steps': m.steps,
                'parameters_schema': m.parameters_schema,
            }
    raise HTTPException(status_code=404, detail=f"Execution type '{slug}' not found")


@router.get("/{slug}/scenarios")
def get_scenarios(slug: str, user=Depends(get_current_user)):
    """Return cherry-picked scenario examples from static fixtures.

    Uses pre-saved trade data from NVDA 5Min with EMA PP V2. The chart bars
    and indicator data are static; the workflow trace is built dynamically
    based on the execution type being viewed.
    """
    from execution_types import list_modules
    from pathlib import Path

    module = None
    for m in list_modules():
        if m.slug == slug:
            module = m
            break
    if not module:
        raise HTTPException(status_code=404, detail=f"Execution type '{slug}' not found")

    exec_code = module.display_code

    # Load static fixture data (cherry-picked trades from NVDA 5Min with EMA PP V2)
    fixture_path = Path(__file__).resolve().parent.parent.parent / "scenario_fixtures.json"
    if not fixture_path.exists():
        return {'slug': slug, 'display_code': exec_code, 'scenarios': [],
                'error': 'Scenario fixtures not found. Run scenario fixture generator.'}

    fixture = json.loads(fixture_path.read_text())

    SCENARIO_META = {
        'stop_loss': {'name': 'Stop Loss Hit', 'description': 'Price entered on EMA 9 cross, then reversed and hit the ATR-based stop loss. Exit at stop level (or bar open if gapped past).'},
        'target': {'name': 'Target Hit', 'description': 'Price entered on EMA 9 cross, then ran to the 2R take profit target. Exit at target level.'},
        'bar_count_exit': {'name': 'Bar Count Exit', 'description': 'Price entered on EMA 9 cross. Position held for 8 bars without hitting stop or target. Exit at bar close.'},
    }

    scenarios = []
    for reason, trade_data in fixture.get('scenarios', {}).items():
        meta = SCENARIO_META.get(reason, {'name': reason.replace('_', ' ').title(), 'description': f'Exit reason: {reason}'})
        ep = trade_data['entry_price']
        xp = trade_data['exit_price']
        sp = trade_data['stop_price']
        tp = trade_data.get('target_price')
        rm = trade_data['r_multiple']
        d = trade_data['direction']
        et = trade_data.get('entry_time', '')
        xt = trade_data.get('exit_time', '')

        markers = [{'time': et, 'position': 'belowBar', 'shape': 'arrowUp', 'color': '#4CAF50', 'text': 'Entry', 'size': 1}]
        xc = '#4CAF50' if rm >= 0 else '#F44336'
        if xt:
            markers.append({'time': xt, 'position': 'aboveBar', 'shape': 'arrowDown', 'color': xc, 'text': f'{rm:+.1f}R', 'size': 1})

        # Dynamic workflow trace based on THIS execution type
        ws = []
        if exec_code in ('C', 'CC'):
            ws.append({'action': 'check_trigger', 'label': 'Bar-close trigger detected', 'time': et, 'badge': exec_code})
        else:
            ws.append({'action': 'check_level_cross', 'label': 'Level cross detected within bar', 'time': et, 'badge': exec_code})
        ws.append({'action': 'fill', 'label': f'Entry filled at ${ep:.2f}', 'price': ep, 'color': 'var(--green)'})
        ws.append({'action': 'fire_webhook', 'label': f'Webhook: entry_{d.lower()}_market', 'isWebhook': True})
        if exec_code in ('LC', 'CC'):
            ws.append({'action': 'wait_confirm', 'label': f'Wait for {"next bar" if exec_code == "CC" else "bar"} close confirmation'})
            ws.append({'action': 'confirmed', 'label': 'Confirmed — position continues', 'color': 'var(--green)'})
        ws.append({'action': 'manage_position', 'label': f'Position open — stop ${sp:.2f}' + (f', target ${tp:.2f}' if tp else '')})
        ws.append({'action': 'exit', 'label': f'{meta["name"]} at ${xp:.2f} ({rm:+.2f}R)', 'time': xt, 'price': xp, 'color': 'var(--green)' if rm >= 0 else 'var(--red)'})
        ws.append({'action': 'fire_webhook', 'label': f'Webhook: exit_{d.lower()}_market', 'isWebhook': True})

        scenarios.append({
            'id': reason, 'name': meta['name'], 'description': meta['description'],
            'direction': d, 'entry_price': ep, 'exit_price': xp,
            'stop_price': sp, 'target_price': tp, 'exit_reason': reason,
            'r_multiple': rm, 'passed': True,
            'chart_bars': trade_data['chart_bars'],
            'ema_data': trade_data.get('ema_data', []),
            'ema_label': fixture.get('indicator', 'EMA 9'),
            'markers': markers, 'workflow_steps': ws,
            'entry_drill': trade_data.get('entry_drill', []),
            'exit_drill': trade_data.get('exit_drill', []),
            'entry_markers': [{'time': et, 'position': 'belowBar', 'shape': 'arrowUp', 'color': '#4CAF50', 'text': 'Entry', 'size': 1}],
            'exit_markers': [{'time': xt, 'position': 'aboveBar', 'shape': 'arrowDown', 'color': xc, 'text': f'{rm:+.1f}R', 'size': 1}] if xt else [],
        })

    return {
        'slug': slug, 'display_code': exec_code,
        'symbol': fixture.get('symbol', 'NVDA'),
        'indicator': fixture.get('indicator', 'EMA 9'),
        'pack': fixture.get('pack', 'EMA Price Position V2'),
        'scenarios': scenarios,
    }

    # === OLD DYNAMIC CODE BELOW (kept as reference, never reached) ===
    try:
        from services import prepare_data_with_indicators, unified_trades
        from confluence_groups import get_entry_triggers, get_enabled_groups

        groups = get_enabled_groups()
        triggers = get_entry_triggers('LONG', groups)
        if not triggers:
            return {'slug': slug, 'display_code': exec_code, 'scenarios': []}

        # Pick first available LONG trigger
        entry_trigger_id = next(iter(triggers))

        df = prepare_data_with_indicators('NVDA', days=30, timeframe='5Min')
        if len(df) < 20:
            return {'slug': slug, 'display_code': exec_code, 'scenarios': []}

        # Use bar_count_exit to get bar count exit scenarios too
        strategy = {
            'symbol': 'NVDA', 'direction': 'LONG', 'timeframe': '5Min',
            'entry_trigger_confluence_id': entry_trigger_id,
            'stop_config': {'method': 'atr', 'atr_mult': 1.5},
            'target_config': {'method': 'risk_reward', 'rr_ratio': 2.0},
            'bar_count_exit': 8,
        }

        # Find EMA column for overlay
        ema_col = None
        for col in ['ema_9', 'ema_21', 'ema_50']:
            if col in df.columns:
                ema_col = col
                break
        trades_df = unified_trades(df, strategy)
    except Exception as e:
        return {'slug': slug, 'display_code': exec_code, 'scenarios': [],
                'error': f'Backtest failed: {e}'}

    if not isinstance(trades_df, pd.DataFrame) or len(trades_df) == 0:
        return {'slug': slug, 'display_code': exec_code, 'scenarios': []}

    # Categorize trades by exit reason and pick one example of each
    SCENARIO_MAP = {
        'stop_loss': {'name': 'Stop Loss Hit', 'description': 'Price entered, then reversed and hit the stop loss. Exit at stop level (or bar open if gapped past).'},
        'target': {'name': 'Target Hit', 'description': 'Price entered, then ran to the take profit target. Exit at target level.'},
        'bar_count_exit': {'name': 'Bar Count Exit', 'description': 'Position held for the maximum number of bars. Exit at bar close.'},
    }
    # Add signal exit (any exit reason that's a trigger ID)
    seen_reasons = set()
    scenarios = []

    for _, trade in trades_df.iterrows():
        reason = trade.get('exit_reason', '')
        if not reason or reason in seen_reasons:
            continue

        # Map to scenario type
        scenario_type = reason if reason in SCENARIO_MAP else 'signal_exit'
        if scenario_type == 'signal_exit' and 'signal_exit' in seen_reasons:
            continue

        seen_reasons.add(reason if reason in SCENARIO_MAP else 'signal_exit')

        entry_time = trade.get('entry_time')
        exit_time = trade.get('exit_time')
        entry_price = float(trade.get('entry_price', 0))
        exit_price = float(trade.get('exit_price', 0))
        r_mult = float(trade.get('r_multiple', 0))
        stop_price = float(trade.get('initial_stop_price', 0))
        target_price = float(trade.get('target_price', 0)) if trade.get('target_price') else None

        # Get chart bars around this trade (15 before entry, 5 after exit)
        try:
            entry_idx = df.index.get_indexer([pd.Timestamp(entry_time)], method='nearest')[0]
            exit_idx = df.index.get_indexer([pd.Timestamp(exit_time)], method='nearest')[0] if exit_time else entry_idx + 10
        except Exception:
            continue

        chart_start = max(0, entry_idx - 15)
        chart_end = min(len(df), exit_idx + 5)
        chart_bars = []
        ema_data = []
        for i in range(chart_start, chart_end):
            row = df.iloc[i]
            chart_bars.append({
                'timestamp': str(df.index[i]),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
            })
            if ema_col and pd.notna(row.get(ema_col)):
                ema_data.append({'time': str(df.index[i]), 'value': float(row[ema_col])})

        # Entry/exit drill-down bars (5 bars around each event)
        entry_drill = []
        exit_drill = []
        for i in range(max(0, entry_idx - 3), min(len(df), entry_idx + 3)):
            row = df.iloc[i]
            entry_drill.append({
                'timestamp': str(df.index[i]),
                'open': float(row['open']), 'high': float(row['high']),
                'low': float(row['low']), 'close': float(row['close']),
            })
        if exit_time:
            for i in range(max(0, exit_idx - 3), min(len(df), exit_idx + 3)):
                row = df.iloc[i]
                exit_drill.append({
                    'timestamp': str(df.index[i]),
                    'open': float(row['open']), 'high': float(row['high']),
                    'low': float(row['low']), 'close': float(row['close']),
                })

        # Build markers
        markers = [
            {'time': str(entry_time), 'position': 'belowBar', 'shape': 'arrowUp', 'color': '#4CAF50', 'text': 'Entry', 'size': 1},
        ]
        if exit_time:
            exit_color = '#4CAF50' if r_mult >= 0 else '#F44336'
            if reason == 'stop_loss':
                exit_color = '#F44336'
            elif reason == 'target':
                exit_color = '#4CAF50'
            markers.append({
                'time': str(exit_time), 'position': 'aboveBar', 'shape': 'arrowDown',
                'color': exit_color, 'text': f'{r_mult:+.1f}R', 'size': 1,
            })

        # Build workflow trace for this trade
        workflow_steps = []
        direction = trade.get('direction', 'LONG')
        et = trade.get('exec_type', 'C')

        # Entry steps
        if et in ('C', 'CC'):
            workflow_steps.append({'action': 'check_trigger', 'label': 'Bar-close trigger detected', 'time': str(entry_time), 'badge': et})
        else:
            workflow_steps.append({'action': 'check_level_cross', 'label': 'Level cross detected', 'time': str(entry_time), 'badge': et})

        workflow_steps.append({'action': 'fill', 'label': f'Entry filled at ${entry_price:.2f}', 'price': entry_price, 'color': 'var(--green)'})
        workflow_steps.append({'action': 'fire_webhook', 'label': f'Webhook: entry_{direction.lower()}_market', 'isWebhook': True})
        workflow_steps.append({'action': 'manage_position', 'label': f'Position open — stop ${stop_price:.2f}' + (f', target ${target_price:.2f}' if target_price else '')})

        # Exit steps
        exit_label = SCENARIO_MAP.get(reason, {}).get('name', f'Exit: {reason.replace("_", " ")}')
        workflow_steps.append({
            'action': 'exit', 'label': f'{exit_label} at ${exit_price:.2f} ({r_mult:+.2f}R)',
            'time': str(exit_time) if exit_time else None, 'price': exit_price,
            'color': 'var(--green)' if r_mult >= 0 else 'var(--red)',
        })
        workflow_steps.append({'action': 'fire_webhook', 'label': f'Webhook: exit_{direction.lower()}_market', 'isWebhook': True})

        meta = SCENARIO_MAP.get(reason, {'name': f'Signal Exit ({reason})', 'description': f'Exit triggered by signal: {reason.replace("_", " ")}'})
        scenarios.append({
            'id': reason if reason in SCENARIO_MAP else 'signal_exit',
            'name': meta['name'],
            'description': meta['description'],
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'exit_reason': reason,
            'r_multiple': round(r_mult, 2),
            'passed': True,
            'chart_bars': chart_bars,
            'ema_data': ema_data,
            'ema_label': ema_col.upper().replace('_', ' ') if ema_col else None,
            'markers': markers,
            'workflow_steps': workflow_steps,
            'entry_drill': entry_drill,
            'exit_drill': exit_drill,
            'entry_markers': [{'time': str(entry_time), 'position': 'belowBar', 'shape': 'arrowUp', 'color': '#4CAF50', 'text': 'Entry', 'size': 1}],
            'exit_markers': [{'time': str(exit_time), 'position': 'aboveBar', 'shape': 'arrowDown', 'color': '#F44336' if r_mult < 0 else '#4CAF50', 'text': f'{r_mult:+.1f}R', 'size': 1}] if exit_time else [],
        })

        if len(scenarios) >= 5:
            break

    return {
        'slug': slug,
        'display_code': exec_code,
        'scenarios': scenarios,
    }


@router.post("/{slug}/simulate")
def simulate_execution_type(
    slug: str,
    body: dict = Body(...),
    user=Depends(get_current_user),
):
    """Simulate an execution type on real market data.

    Loads recent bars, finds where a simple EMA cross trigger fires,
    then traces the execution type's behavior step by step.

    Body: {symbol, timeframe, direction, days}
    Returns: step-by-step trace of what the execution type did.
    """
    from execution_types import list_modules, get_module
    import pandas as pd

    # Find the module
    module = None
    for m in list_modules():
        if m.slug == slug:
            module = m
            break
    if not module:
        raise HTTPException(status_code=404, detail=f"Execution type '{slug}' not found")

    symbol = body.get('symbol', 'NVDA')
    timeframe = body.get('timeframe', '5Min')
    direction = body.get('direction', 'LONG')
    days = body.get('days', 14)

    # Load market data
    try:
        from services import prepare_data_with_indicators
        df = prepare_data_with_indicators(symbol, days=days, timeframe=timeframe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load data: {e}")

    if len(df) < 20:
        raise HTTPException(status_code=400, detail="Not enough data bars")

    # Find a bar where a simple trigger fires (EMA short cross above close)
    # Use a basic cross: close crosses above EMA-21 (or first available EMA)
    ema_col = None
    for col in ['ema_21', 'ema_9', 'ema_50']:
        if col in df.columns:
            ema_col = col
            break

    trigger_bar_idx = None
    if ema_col:
        for i in range(1, len(df)):
            prev_close = float(df.iloc[i-1]['close'])
            curr_close = float(df.iloc[i]['close'])
            prev_ema = float(df.iloc[i-1].get(ema_col, 0))
            curr_ema = float(df.iloc[i].get(ema_col, 0))
            if prev_ema > 0 and curr_ema > 0:
                if direction == 'LONG':
                    if prev_close <= prev_ema and curr_close > curr_ema:
                        trigger_bar_idx = i
                        break
                else:
                    if prev_close >= prev_ema and curr_close < curr_ema:
                        trigger_bar_idx = i
                        break

    if trigger_bar_idx is None:
        # Fallback: just pick a bar in the middle
        trigger_bar_idx = len(df) // 2

    trigger_bar = df.iloc[trigger_bar_idx]
    trigger_time = str(df.index[trigger_bar_idx])
    close_price = float(trigger_bar['close'])
    atr = float(trigger_bar.get('atr', close_price * 0.01))
    if atr <= 0:
        atr = close_price * 0.01

    # Compute stop/target
    stop_distance = atr * 1.5
    stop_price = close_price - stop_distance if direction == 'LONG' else close_price + stop_distance
    target_price = close_price + (stop_distance * 2) if direction == 'LONG' else close_price - (stop_distance * 2)

    # Build execution trace
    exec_code = module.display_code
    trace_steps = []

    # Step 1: Trigger fires
    if exec_code in ('C', 'CC'):
        trace_steps.append({
            'bar': trigger_bar_idx,
            'time': trigger_time,
            'action': 'trigger_fired',
            'label': f'Bar-close trigger fired at {trigger_time}',
            'price': close_price,
        })
        fill_price = close_price
    else:
        # L or LC: level cross
        level_price = float(trigger_bar.get(ema_col, close_price)) if ema_col else close_price
        trace_steps.append({
            'bar': trigger_bar_idx,
            'time': trigger_time,
            'action': 'trigger_fired',
            'label': f'Level cross detected at {trigger_time}',
            'price': level_price,
        })
        fill_price = level_price

    # Step 2: Fill
    trace_steps.append({
        'bar': trigger_bar_idx,
        'time': trigger_time,
        'action': 'fill',
        'label': f'Filled at ${fill_price:.2f}',
        'price': fill_price,
    })

    # Step 3: Webhook
    webhook_event = f'entry_{direction.lower()}_market'
    trace_steps.append({
        'bar': trigger_bar_idx,
        'time': trigger_time,
        'action': 'fire_webhook',
        'label': f'Webhook: {webhook_event}',
        'event': webhook_event,
    })

    # Step 4: Confirmation (LC/CC only)
    confirmed = True
    if exec_code in ('LC', 'CC'):
        confirm_offset = 1 if exec_code == 'CC' else 0
        confirm_bar_idx = trigger_bar_idx + confirm_offset
        if confirm_bar_idx < len(df):
            confirm_bar = df.iloc[confirm_bar_idx]
            confirm_close = float(confirm_bar['close'])
            confirm_time = str(df.index[confirm_bar_idx])

            # Simple confirmation: close is on the right side of the level
            if direction == 'LONG':
                confirmed = confirm_close > fill_price
            else:
                confirmed = confirm_close < fill_price

            trace_steps.append({
                'bar': confirm_bar_idx,
                'time': confirm_time,
                'action': 'confirmation_check',
                'label': f'Confirmation check at {confirm_time}: {"CONFIRMED" if confirmed else "NOT CONFIRMED"}',
                'confirmed': confirmed,
                'price': confirm_close,
            })

            if not confirmed:
                bail_event = f'exit_{direction.lower()}_market'
                trace_steps.append({
                    'bar': confirm_bar_idx,
                    'time': confirm_time,
                    'action': 'bail',
                    'label': f'Bail: exit at market (${confirm_close:.2f})',
                    'price': confirm_close,
                })
                trace_steps.append({
                    'bar': confirm_bar_idx,
                    'time': confirm_time,
                    'action': 'fire_webhook',
                    'label': f'Webhook: {bail_event} (bail)',
                    'event': bail_event,
                })

    # Step 5: Position management (if confirmed)
    if confirmed:
        trace_steps.append({
            'bar': trigger_bar_idx,
            'time': trigger_time,
            'action': 'position_open',
            'label': f'Position open: stop=${stop_price:.2f}, target=${target_price:.2f}',
            'stop_price': stop_price,
            'target_price': target_price,
        })

        # Scan forward for exit
        exit_bar_idx = None
        exit_price = None
        exit_reason = None
        start = trigger_bar_idx + (2 if exec_code == 'CC' else 1)
        for i in range(start, min(len(df), trigger_bar_idx + 20)):
            bar = df.iloc[i]
            high = float(bar['high'])
            low = float(bar['low'])
            bar_close = float(bar['close'])

            if direction == 'LONG':
                if low <= stop_price:
                    exit_bar_idx = i
                    exit_price = stop_price
                    exit_reason = 'stop_loss'
                    break
                if high >= target_price:
                    exit_bar_idx = i
                    exit_price = target_price
                    exit_reason = 'target'
                    break
            else:
                if high >= stop_price:
                    exit_bar_idx = i
                    exit_price = stop_price
                    exit_reason = 'stop_loss'
                    break
                if low <= target_price:
                    exit_bar_idx = i
                    exit_price = target_price
                    exit_reason = 'target'
                    break

        if exit_bar_idx is None:
            exit_bar_idx = min(len(df) - 1, trigger_bar_idx + 19)
            exit_price = float(df.iloc[exit_bar_idx]['close'])
            exit_reason = 'end_of_data'

        exit_time = str(df.index[exit_bar_idx])
        pnl = (exit_price - fill_price) if direction == 'LONG' else (fill_price - exit_price)
        risk = abs(fill_price - stop_price)
        r_multiple = pnl / risk if risk > 0 else 0

        trace_steps.append({
            'bar': exit_bar_idx,
            'time': exit_time,
            'action': 'exit',
            'label': f'Exit: {exit_reason} at ${exit_price:.2f} ({r_multiple:+.2f}R)',
            'price': exit_price,
            'reason': exit_reason,
            'r_multiple': round(r_multiple, 2),
        })
        trace_steps.append({
            'bar': exit_bar_idx,
            'time': exit_time,
            'action': 'fire_webhook',
            'label': f'Webhook: exit_{direction.lower()}_market',
            'event': f'exit_{direction.lower()}_market',
        })

    # Build chart data window (20 bars before trigger, 20 after exit)
    chart_start = max(0, trigger_bar_idx - 20)
    chart_end_idx = trigger_bar_idx + 20
    for s in trace_steps:
        if s.get('bar') and s['bar'] > chart_end_idx - 5:
            chart_end_idx = s['bar'] + 10
    chart_end = min(len(df), chart_end_idx)

    chart_bars = []
    for i in range(chart_start, chart_end):
        row = df.iloc[i]
        chart_bars.append({
            'timestamp': str(df.index[i]),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
        })

    # Build trade markers for the chart
    chart_markers = []
    for s in trace_steps:
        if s['action'] == 'fill':
            chart_markers.append({
                'time': s['time'], 'position': 'belowBar' if direction == 'LONG' else 'aboveBar',
                'shape': 'arrowUp' if direction == 'LONG' else 'arrowDown',
                'color': '#4CAF50', 'text': 'Entry', 'size': 1,
            })
        elif s['action'] == 'exit':
            color = '#4CAF50' if s.get('r_multiple', 0) >= 0 else '#F44336'
            chart_markers.append({
                'time': s['time'], 'position': 'aboveBar' if direction == 'LONG' else 'belowBar',
                'shape': 'arrowDown', 'color': color,
                'text': f'{s.get("r_multiple", 0):+.1f}R', 'size': 1,
            })
        elif s['action'] == 'bail':
            chart_markers.append({
                'time': s['time'], 'position': 'aboveBar' if direction == 'LONG' else 'belowBar',
                'shape': 'arrowDown', 'color': '#FF9800',
                'text': 'Bail', 'size': 1,
            })

    return {
        'slug': slug,
        'display_code': module.display_code,
        'symbol': symbol,
        'timeframe': timeframe,
        'direction': direction,
        'trigger_bar': trigger_bar_idx,
        'trigger_time': trigger_time,
        'fill_price': fill_price,
        'stop_price': stop_price if confirmed else None,
        'target_price': target_price if confirmed else None,
        'confirmed': confirmed,
        'steps': trace_steps,
        'chart_bars': chart_bars,
        'chart_markers': chart_markers,
    }


# ---- Storage helpers ----

def _load_config() -> dict:
    from db import USE_DB
    if USE_DB:
        from db import load_settings_db
        settings = load_settings_db()
        return settings.get('execution_types', {})
    return {}


def _save_config(config: dict):
    from db import USE_DB
    if USE_DB:
        from db import load_settings_db, save_settings_db
        settings = load_settings_db()
        settings['execution_types'] = config
        save_settings_db(settings)
