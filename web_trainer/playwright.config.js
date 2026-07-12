import { defineConfig } from '@playwright/test';

export default defineConfig({
    testDir: './tests',
    timeout: 60_000,
    use: {
        headless: true,
        baseURL: 'http://localhost:8765',
    },
    webServer: {
        command: 'python3 -m http.server 8765',
        cwd: '../docs',
        url: 'http://localhost:8765',
        reuseExistingServer: true,
        timeout: 10_000,
    },
});
