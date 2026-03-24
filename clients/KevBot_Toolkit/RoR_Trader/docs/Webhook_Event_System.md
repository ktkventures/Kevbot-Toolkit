# Webhook Event System — Reference Specification

> Version 1.0 — 2026-03-24
> Status: DESIGN APPROVED, implementation pending

---

## 1. Overview

Webhooks are **actionable instructions sent to an exchange or execution service** (e.g., SignalStack, TradeThePool). The system only fires a webhook when it wants the recipient to take immediate action. All internal logic (hold times, confirmations, gate checks) is resolved before any webhook leaves the system.

### Core Principle
> If no action is needed on the exchange, no webhook is sent.

---

## 2. Webhook Event Types

The system defines **11 webhook event types**. Each represents a discrete instruction to the exchange.

| # | Event Type | Direction | Order Type | Description |
|---|---|---|---|---|
| 1 | `entry_long_market` | LONG | Market | Open a long position at market price |
| 2 | `entry_long_limit` | LONG | Limit | Open a long position at a specific limit price |
| 3 | `entry_short_market` | SHORT | Market | Open a short position at market price |
| 4 | `entry_short_limit` | SHORT | Limit | Open a short position at a specific limit price |
| 5 | `exit_long_market` | LONG | Market | Close a long position at market price |
| 6 | `exit_long_limit` | LONG | Limit | Close a long position at a specific limit price |
| 7 | `exit_short_market` | SHORT | Market | Close a short position at market price |
| 8 | `exit_short_limit` | SHORT | Limit | Close a short position at a specific limit price |
| 9 | `cancel_long` | LONG | — | Cancel an unfilled long limit order |
| 10 | `cancel_short` | SHORT | — | Cancel an unfilled short limit order |
| 11 | `compliance_breach` | — | — | Portfolio requirement rule violated |

---

## 3. Execution Type → Webhook Event Mapping

Each execution type produces a specific sequence of webhook events depending on conditions.

### 3.1 [C] Close — Bar Close Entry

**Mechanics:** Entry fires when the bar closes and the trigger evaluates TRUE. Always a market order.

**Webhook sequence:**
```
1. entry_X_market        ← fires at bar close
   ... position is open ...
2. exit_X_market          ← fires on exit (signal, stop, target, bar count)
```

**Timing:** Both webhooks fire immediately when the condition is met. No delay.

---

### 3.2 [L] Level — Intra-Bar Level Cross Entry

**Mechanics:** Entry fires when price crosses an indicator level intra-bar. The pack's `reference_bar` parameter determines whether the level comes from the current bar (0) or previous bar (-1). The `order_type` parameter determines market or limit.

#### If configured as MARKET order:
```
1. entry_X_market        ← fires when level crossed (after hold_seconds if configured)
   ... position is open ...
2. exit_X_market          ← fires on stop loss, exit trigger, target, or bar count
```

#### If configured as LIMIT order:
```
1. entry_X_limit         ← fires when level crossed, limit at level price
   ... waiting for fill ...
   IF not filled within limit_duration_seconds:
2. cancel_X              ← fires to cancel unfilled limit
```

**Hold seconds:** If `hold_seconds > 0`, the system waits internally for the condition to hold for the specified duration. The webhook only fires after the hold completes. The exchange never sees the hold — it only receives the final order.

---

### 3.3 [LC] Level-Close — Two-Stage: Level Cross Entry + Bar Close Confirmation

**Mechanics:** Two-stage execution. First letter (L) = how you get in (level cross). Second letter (C) = how you confirm (bar close). The pack's `entry_order_type` determines whether entry is market or limit.

#### Normal flow (confirmed):
```
1. entry_X_market/limit  ← 1st trade: fires at level cross
   ... bar continues forming ...
   Bar closes → confirmation check PASSES
   ... position remains open ...
2. exit_X_market          ← fires on stop loss, exit trigger, target, or bar count
```

#### Unconfirmed flow (bail out, market entry):
```
1. entry_X_market        ← 1st trade: fires at level cross
   ... bar continues forming ...
   Bar closes → confirmation check FAILS
2. exit_X_market          ← 2nd trade: fires immediately at bar close (bail out)
```

#### Unconfirmed flow (bail out, limit entry):
```
1. entry_X_limit         ← 1st trade: fires at level cross, limit at level price
   ... bar continues forming ...
   Bar closes → confirmation check FAILS
2. exit_X_limit           ← 2nd trade: fires at entry price (exit via limit to minimize slippage)
```

#### Limit timeout flow (limit entry never fills):
```
1. entry_X_limit         ← 1st trade: fires at level cross
   ... limit_duration_seconds elapses, no fill ...
2. cancel_X              ← fires to cancel unfilled limit
   (optional fallback)
3. entry_X_market         ← fires as market fallback if still desired
```

**Key behavior:** The entry webhook fires at the level cross (not at bar close). If confirmation fails, the exit fires at bar close — the position was open for less than one bar.

---

### 3.4 [CC] Close-Close — Two-Stage: Bar Close Entry + Next Bar Close Confirmation

**Mechanics:** Two-stage execution. First letter (C) = how you get in (bar close). Second letter (C) = how you confirm (next bar close). Entry is always a market order.

#### Normal flow (confirmed):
```
1. entry_X_market        ← 1st trade: fires at bar close
   ... next bar forms ...
   Next bar closes → confirmation check PASSES
   ... position remains open ...
2. exit_X_market          ← fires on stop loss, exit trigger, target, or bar count
```

#### Unconfirmed flow (bail out):
```
1. entry_X_market        ← 1st trade: fires at bar close
   ... next bar forms ...
   Next bar closes → confirmation check FAILS
2. exit_X_market          ← 2nd trade: fires at next bar close (bail out)
```

**Key behavior:** Position is held for at least one full bar while waiting for confirmation. If the next bar close doesn't confirm, exit fires immediately.

---

## 4. Exit Event Mapping

Exits produce webhooks based on the exit reason:

| Exit Reason | Webhook Event | Notes |
|---|---|---|
| `exit_trigger` | `exit_X_market` | Opposite signal or exit trigger fires at bar close |
| `stop_loss` | `exit_X_market` | Stop price breached (intra-bar level cross) |
| `target` | `exit_X_market` | Take profit level reached |
| `bar_count_exit` | `exit_X_market` | Maximum hold time exceeded |
| `unconfirmed_lc` | `exit_X_market` or `exit_X_limit` | [LC] confirmation failed — market if entry was market, limit at entry price if entry was limit |
| `unconfirmed_cc` | `exit_X_market` | [CC] confirmation failed, bail at market on next bar close |

Where `X` = `long` or `short` based on strategy direction.

---

## 5. Available Payload Placeholders

These variables are available in webhook JSON payload templates:

### Signal Context
| Placeholder | Type | Example | Description |
|---|---|---|---|
| `{{symbol}}` | string | `SPY` | Ticker symbol |
| `{{direction}}` | string | `LONG` | Strategy direction |
| `{{timeframe}}` | string | `1Min` | Bar timeframe |
| `{{strategy_name}}` | string | `NVDA LONG - Mass #2` | Strategy display name |
| `{{strategy_id}}` | string | `5` | Strategy numeric ID |
| `{{event_type}}` | string | `entry_long_market` | Webhook event type (from Section 2) |
| `{{timestamp}}` | string | `2026-03-24T14:30:00Z` | ISO timestamp of signal |

### Order Context
| Placeholder | Type | Example | Description |
|---|---|---|---|
| `{{order_action}}` | string | `buy` | `buy`, `sell`, or `close` |
| `{{order_type}}` | string | `market` | `market` or `limit` |
| `{{order_price}}` | number | `142.35` | Fill price (market) or limit price |
| `{{stop_price}}` | number | `141.50` | Calculated stop loss price |
| `{{quantity}}` | number | `10` | Share quantity (derived from portfolio risk / stop distance) |
| `{{market_position}}` | string | `long` | `long`, `short`, or `flat` |

### Indicator Context
| Placeholder | Type | Example | Description |
|---|---|---|---|
| `{{trigger_name}}` | string | `EMA_CROSS` | Trigger ID that fired |
| `{{atr}}` | number | `1.50` | ATR value at signal bar |
| `{{confluence_met}}` | string | `1M-EMA_STACK-LML, 15M-MACD_LINE-AB` | Active confluence conditions |

### Portfolio Context
| Placeholder | Type | Example | Description |
|---|---|---|---|
| `{{portfolio_name}}` | string | `Main Portfolio` | Portfolio display name |
| `{{portfolio_id}}` | string | `1` | Portfolio numeric ID |
| `{{risk_per_trade}}` | number | `500.00` | Dollar risk per trade |
| `{{position_risk}}` | number | `500.00` | Position risk in this portfolio |

### Compliance Context (compliance_breach only)
| Placeholder | Type | Example | Description |
|---|---|---|---|
| `{{rule_name}}` | string | `Daily Loss Limit` | Breached rule name |
| `{{rule_limit}}` | string | `-$5000` | Rule limit display |
| `{{rule_value}}` | string | `-$5200` | Actual value display |

---

## 6. Account-Based Webhook Templates

### Concept

Instead of configuring individual JSON payloads per webhook per portfolio, users create one **Account Template** per exchange/service. The template defines JSON payloads for all 11 event types. Portfolios simply select which template to use.

### Template Structure

```
Account Template: "SignalStack - Main Account"
├── entry_long_market:    { "action": "buy",  "type": "market", "symbol": "{{symbol}}", "qty": {{quantity}} }
├── entry_long_limit:     { "action": "buy",  "type": "limit",  "symbol": "{{symbol}}", "qty": {{quantity}}, "price": {{order_price}} }
├── entry_short_market:   { "action": "sell", "type": "market", "symbol": "{{symbol}}", "qty": {{quantity}} }
├── entry_short_limit:    { "action": "sell", "type": "limit",  "symbol": "{{symbol}}", "qty": {{quantity}}, "price": {{order_price}} }
├── exit_long_market:     { "action": "sell", "type": "market", "symbol": "{{symbol}}", "qty": {{quantity}} }
├── exit_long_limit:      { "action": "sell", "type": "limit",  "symbol": "{{symbol}}", "qty": {{quantity}}, "price": {{order_price}} }
├── exit_short_market:    { "action": "buy",  "type": "market", "symbol": "{{symbol}}", "qty": {{quantity}} }
├── exit_short_limit:     { "action": "buy",  "type": "limit",  "symbol": "{{symbol}}", "qty": {{quantity}}, "price": {{order_price}} }
├── cancel_long:          { "action": "cancel", "symbol": "{{symbol}}", "side": "buy" }
├── cancel_short:         { "action": "cancel", "symbol": "{{symbol}}", "side": "sell" }
└── compliance_breach:    { "alert": "compliance", "portfolio": "{{portfolio_name}}", "rule": "{{rule_name}}" }
```

### Why This Approach
- **One-time setup per exchange.** JSON payloads differ by exchange but event types are universal.
- **Portfolio just picks a template.** No manual wiring of individual webhooks.
- **Clear separation of concerns.** Strategy defines *what* triggers. Template defines *how* to communicate with the exchange. Portfolio defines *how much* (risk/quantity).

### Data Flow
```
Strategy (triggers, execution type)
  → Alert Engine (detects signal, resolves execution params)
    → Portfolio (adds risk_per_trade, quantity, selects template)
      → Template (resolves event type → JSON payload)
        → Webhook delivery (HTTP POST to exchange URL)
```

---

## 7. Relationship Between Strategy, Portfolio, and Webhooks

| Concern | Defined By | Example |
|---|---|---|
| What triggers entry/exit | Strategy (pack + trigger selection) | [C] EMA Short > Mid Cross |
| Execution type (market/limit) | Strategy (pack execution type config) | [HL] = limit entry |
| Which webhook events fire | Derived from execution type (this spec) | entry_long_limit → cancel_long |
| Risk per trade / quantity | Portfolio | $500 risk → 10 shares |
| JSON payload format | Account Template | SignalStack format |
| Webhook URL | Account Template | https://api.signalstack.com/... |
| Event filtering | Portfolio (which events to subscribe to) | Enable entry_long_market, disable compliance |

---

## 8. Migration from Current System

### Current (5 events)
```
entry_long, entry_short, exit_long, exit_short, compliance_breach
```

### New (11 events)
The old events map to the new system:
- `entry_long` → `entry_long_market` (default) or `entry_long_limit`
- `entry_short` → `entry_short_market` (default) or `entry_short_limit`
- `exit_long` → `exit_long_market` (default) or `exit_long_limit`
- `exit_short` → `exit_short_market` (default) or `exit_short_limit`
- `compliance_breach` → `compliance_breach` (unchanged)
- New: `cancel_long`, `cancel_short`

Existing webhook configs using old event names will auto-map to `_market` variants for backward compatibility.
