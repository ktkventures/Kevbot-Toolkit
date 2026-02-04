# RoR Trader - Wireframes Overview

**Version:** 0.1
**Date:** February 3, 2026

---

## Design Principles

Based on the Trade Analyzer patterns, the RoR Trader UI will follow these principles:

1. **Data Transparency** - Always show the underlying trades, not just aggregated KPIs
2. **Progressive Disclosure** - Core workflow front and center, details in expandable sections
3. **Real-time Feedback** - Show impact of adding/removing confluence immediately
4. **Clean Information Hierarchy** - KPIs at top, charts in middle, details below

---

## Screen Index

| Screen | File | Priority | Description |
|--------|------|----------|-------------|
| Strategy Builder | [01_Strategy_Builder.md](01_Strategy_Builder.md) | P0 | Core 3-step workflow |
| My Strategies | [02_My_Strategies.md](02_My_Strategies.md) | P0 | List and detail views |
| Dashboard | [03_Dashboard.md](03_Dashboard.md) | P1 | Overview and quick actions |
| Portfolios | [04_Portfolios.md](04_Portfolios.md) | P1 | Portfolio builder and analysis |
| Settings | [05_Settings.md](05_Settings.md) | P0 | Interpreters, connections |

---

## Global Navigation

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  [RoR Logo]   Dashboard   Strategies   Portfolios   Alerts   Settings    [User]│
└────────────────────────────────────────────────────────────────────────────────┘
```

- **Logo** - Returns to Dashboard
- **Dashboard** - Overview of all activity
- **Strategies** - My Strategies list + Strategy Builder
- **Portfolios** - Portfolio list + builder
- **Alerts** - Alert history and webhook config
- **Settings** - Interpreters, connections, preferences
- **User** - Profile, logout

---

## Color Palette (Suggested)

| Use | Color | Hex |
|-----|-------|-----|
| Primary | Blue | #2196F3 |
| Success/Profit | Green | #4CAF50 |
| Warning | Orange | #FF9800 |
| Error/Loss | Red | #F44336 |
| Background | Light Gray | #F5F5F5 |
| Card Background | White | #FFFFFF |
| Text Primary | Dark Gray | #212121 |
| Text Secondary | Medium Gray | #757575 |

---

## Common Components

### KPI Card
```
┌─────────────────┐
│ Win Rate        │
│ 62.5%           │
│ ▲ +3.2%         │
└─────────────────┘
```

### Confluence Tag (Selected)
```
┌─────────────────────┐
│ ✕ 1M-EMA-SML       │
└─────────────────────┘
```

### Confluence Row (Selectable)
```
┌───┬────────────────┬──────┬──────┬──────┬────────┐
│ ☐ │ 5M-MACD-M>S↑  │  45  │ 1.8  │ 58%  │ +0.35R │
└───┴────────────────┴──────┴──────┴──────┴────────┘
  ^       ^            ^      ^      ^       ^
Check  Confluence    Trades   PF    WR%   Daily R
```

### Trade Row
```
┌──────────┬───────┬───────┬────────┬─────┬──────────────────────┐
│ 09:45    │   A   │   B   │ +1.25R │  ✓  │ 1M-EMA-SML, 5M-M... │
└──────────┴───────┴───────┴────────┴─────┴──────────────────────┘
    ^         ^       ^        ^       ^            ^
  Time     Entry    Exit      R     Win?     Confluences
```

---

## Responsive Considerations

- **Desktop (1200px+)**: Full layout with side-by-side panels
- **Tablet (768-1199px)**: Stack panels vertically, maintain sidebar
- **Mobile (< 768px)**: Bottom navigation, full-width cards, collapsible sections

---

## Next Steps

1. Review wireframes with stakeholders
2. Create high-fidelity mockups in Figma (optional)
3. Build component library
4. Implement MVP screens
