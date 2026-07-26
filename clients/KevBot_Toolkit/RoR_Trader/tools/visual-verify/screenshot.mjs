#!/usr/bin/env node
// Headless screenshot helper for task-thread visual verification (board #107).
//
// Usage:
//   node screenshot.mjs <path-or-url> <out.png> [options]
//
//   <path-or-url>  App path ("/admin/tasks", "/strategies/303") or a full URL.
//   <out.png>      Output PNG path (created; parent dirs made as needed).
//
// Options:
//   --base <url>     Base URL when arg 1 is a path. Default: dev frontend.
//                    Use --base http://localhost:3000 for local preview.
//   --wait <ms>      Extra settle wait after navigation (default 4000;
//                    strategy-detail chart pages want 12000+).
//   --viewport WxH   Viewport size (default 1440x900).
//   --full           Full-page screenshot instead of viewport crop.
//   --no-login       Skip the login flow (public pages only).
//
// Login creds come from src/.env (ROR_TEST_EMAIL / ROR_TEST_PASSWORD) —
// gitignored, never hardcode them here.

import { chromium } from 'playwright';
import { readFileSync, mkdirSync } from 'fs';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const DEFAULT_BASE = 'https://frontend-dev-e01a.up.railway.app';
const HERE = dirname(fileURLToPath(import.meta.url));
const ENV_PATH = resolve(HERE, '../../src/.env');

function readCreds() {
  const env = readFileSync(ENV_PATH, 'utf8');
  const get = (key) => (env.match(new RegExp(`^${key}=(.*)$`, 'm')) || [])[1]?.trim();
  const email = get('ROR_TEST_EMAIL');
  const password = get('ROR_TEST_PASSWORD');
  if (!email || !password) throw new Error(`ROR_TEST_EMAIL/ROR_TEST_PASSWORD not found in ${ENV_PATH}`);
  return { email, password };
}

function parseArgs(argv) {
  const args = { wait: 4000, viewport: { width: 1440, height: 900 }, full: false, login: true, base: DEFAULT_BASE };
  const positional = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--base') args.base = argv[++i];
    else if (a === '--wait') args.wait = parseInt(argv[++i], 10);
    else if (a === '--viewport') {
      const [w, h] = argv[++i].split('x').map(Number);
      args.viewport = { width: w, height: h };
    } else if (a === '--full') args.full = true;
    else if (a === '--no-login') args.login = false;
    else positional.push(a);
  }
  if (positional.length !== 2) {
    console.error('Usage: node screenshot.mjs <path-or-url> <out.png> [--base url] [--wait ms] [--viewport WxH] [--full] [--no-login]');
    process.exit(2);
  }
  args.target = positional[0].startsWith('http') ? positional[0] : args.base.replace(/\/$/, '') + positional[0];
  args.base = positional[0].startsWith('http') ? new URL(positional[0]).origin : args.base;
  args.out = positional[1];
  return args;
}

const args = parseArgs(process.argv.slice(2));
mkdirSync(dirname(resolve(args.out)), { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: args.viewport });
try {
  if (args.login) {
    const { email, password } = readCreds();
    await page.goto(args.base + '/login', { waitUntil: 'domcontentloaded', timeout: 60000 });
    // Settle past React hydration — a click before handlers attach is silently lost
    // (bites on cold local dev servers; the deployed site hydrates fast enough).
    await page.waitForLoadState('networkidle', { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(1500);
    await page.fill('input[type=email]', email);
    await page.fill('input[type=password]', password);
    // NOT page.click('button[type=submit]') — sidebar nav buttons are also type=submit.
    // Poll for the token; re-click every ~9s in case the first click predated hydration.
    let token = null;
    for (let waited = 0; !token && waited < 45000; waited += 1500) {
      if (waited % 9000 === 0) {
        await page.getByRole('button', { name: 'Sign In', exact: true }).click().catch(() => {});
      }
      await page.waitForTimeout(1500);
      token = await page.evaluate(() => localStorage.getItem('ror_access_token')).catch(() => null);
    }
    if (!token) throw new Error('Login failed: no ror_access_token in localStorage after Sign In');
    console.log('login: OK');
  }
  await page.goto(args.target, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForTimeout(args.wait);
  await page.screenshot({ path: args.out, fullPage: args.full });
  console.log(`screenshot: ${resolve(args.out)} (${args.viewport.width}x${args.viewport.height}${args.full ? ', fullPage' : ''})`);
} finally {
  await browser.close();
}
