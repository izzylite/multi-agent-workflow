---
title: Research Session
query: "Browserbase MCP server documentation setup installation latest version npm package"
date: 7/31/2025
time: 5:32:01 PM
timestamp: 2025-07-31T16:32:01.620Z
exchanges: 1
---

# Research Session

## Initial Query

**Question:** Browserbase MCP server documentation setup installation latest version npm package

**Response:**

The **Browserbase MCP server** enables cloud browser automation via the Model Context Protocol (MCP), allowing LLMs and agent frameworks to control browsers for navigation, interaction, and data extraction. The latest official implementation is distributed as an NPM package, with both hosted and local deployment options. Below is a comprehensive guide to setup, installation, and integration, tailored for your project’s requirements.

---

## 1. Overview and Capabilities

- **Browserbase MCP Server** provides a bridge between LLM-powered agents (like CrewAI) and browser automation, supporting:
  - Natural language browser control
  - Multi-session management
  - Data extraction and screenshot capture
  - Cookie and proxy management
  - Advanced stealth and anti-bot features (for Scale Plan users)[3][1][2]

- **Key Technologies**: Built on top of Stagehand (Browserbase’s AI browser automation SDK), leveraging Playwright under the hood[3][4].

---

## 2. Installation (NPM Package)

### Latest NPM Package

- **Package Name**: `@browserbasehq/mcp-server-browserbase`
- **Latest Version**: Always check [npmjs.com/package/@browserbasehq/mcp-server-browserbase](https://www.npmjs.com/package/@browserbasehq/mcp-server-browserbase) for the most recent release. As of July 2025, the package is actively maintained and updated[4].

### Install via NPM or NPX

```bash
# Using npx (recommended for quick start)
npx @browserbasehq/mcp-server-browserbase

# Or install globally
npm install -g @browserbasehq/mcp-server-browserbase
mcp-server-browserbase
```

- **Environment Variables** (required for API access):
  - `BROWSERBASE_API_KEY`
  - `BROWSERBASE_PROJECT_ID`
  - (Optional) `GEMINI_API_KEY` for Stagehand LLM integration[2]

### Local Development (from Source)

```bash
git clone https://github.com/browserbase/mcp-server-browserbase.git
cd mcp-server-browserbase
pnpm install && pnpm build
```
- Use `pnpm` for dependency management as recommended by the maintainers[2].

---

## 3. Configuration and Usage

### Command-Line Flags

The server supports a range of CLI flags for customization:

| Flag                       | Description                                                      |
|----------------------------|------------------------------------------------------------------|
| `--proxies`                | Enable Browserbase proxies for the session                       |
| `--advancedStealth`        | Enable advanced stealth (Scale Plan only)                        |
| `--contextId <contextId>`  | Specify a Browserbase Context ID                                 |
| `--persist [boolean]`      | Persist the Browserbase context (default: true)                  |
| `--port <port>`            | Port to listen on (HTTP/SHTTP transport)                         |
| `--host <host>`            | Host to bind server (default: localhost, use 0.0.0.0 for all)    |
| `--cookies [json]`         | JSON array of cookies to inject                                  |
| `--browserWidth <width>`   | Browser viewport width (default: 1024)                           |
| `--browserHeight <height>` | Browser viewport height (default: 768)                           |
| `--modelName <model>`      | Model for Stagehand (default: google/gemini-2.0-flash)           |
| `--modelApiKey <key>`      | API key for custom model provider                                |

- Flags can be passed directly to the CLI or set in a configuration file[1].

### Example: Running with Custom Configuration

```bash
BROWSERBASE_API_KEY=your_key \
BROWSERBASE_PROJECT_ID=your_project \
npx @browserbasehq/mcp-server-browserbase \
  --port 8080 \
  --proxies \
  --advancedStealth \
  --browserWidth 1280 \
  --browserHeight 800
```

---

## 4. Deployment Options

- **Remote Hosted**: Use Browserbase’s hosted MCP server by specifying the remote URL in your agent configuration. This is the fastest way to get started and is recommended for most production use cases[2][3].
- **Local Server**: For full control or development, run the MCP server locally as described above. This is useful for debugging, custom integrations, or when you need to inject custom cookies, proxies, or advanced settings[2].

---

## 5. Integration with Your Project

Given your project context (CrewAI tools, session pooling, vendor-specific agents):

- **Session Management**: The MCP server supports multi-session workflows, which aligns with your `BrowserbaseManager` and session pooling requirements[3].
- **Configuration**: Use CLI flags or environment variables to set user agents, proxies, and viewport sizes as needed for different vendors (Tesco, Asda, Costco).
- **Tool Wrappers**: Your `BrowserbaseTool` classes can interact with the MCP server via HTTP/SHTTP endpoints, sending natural language or structured commands for navigation, interaction, and extraction.
- **Testing**: For unit and integration tests, you can mock MCP server responses or run a local instance with test credentials.

---

## 6. Advanced Features and Updates

- **Stealth and Anti-Bot**: Enable `--advancedStealth` for enhanced anti-bot evasion (Scale Plan required)[1].
- **CAPTCHA Handling**: The latest SDKs (Node v2.6.0, Python v1.4.0) support custom selectors for CAPTCHA image and input fields, which is critical for scraping sites like Tesco[4].
- **Keep-Alive Sessions**: Improved reliability for long-running sessions with the `keepAlive` option, ensuring sessions persist across agent operations[4].
- **Model Integration**: Supports integration with Gemini and other LLMs for natural language automation[1][3].

---

## 7. Documentation and Further Reading

- **Official Docs**: [docs.browserbase.com/integrations/mcp/introduction](https://docs.browserbase.com/integrations/mcp/introduction)[3]
- **GitHub Source**: [github.com/browserbase/mcp-server-browserbase](https://github.com/browserbase/mcp-server-browserbase)[1]
- **Changelog**: [browserbase.com/changelog](https://www.browserbase.com/changelog)[4]

---

## 8. Edge Cases and Pitfalls

- **API Keys**: Ensure all required API keys are set; missing keys will prevent server startup or session creation[2].
- **Plan Restrictions**: Some features (e.g., advanced stealth) are only available on higher-tier plans[1].
- **Port Conflicts**: When running locally, ensure the chosen port is not in use by other services[1].
- **Session Limits**: Your Browserbase plan may limit concurrent sessions; implement pooling and cleanup logic in your `BrowserbaseManager`[3].
- **Vendor-Specific Challenges**: For sites with aggressive anti-bot measures (e.g., Tesco), leverage the latest SDK features for CAPTCHA and stealth[4].

---

## 9. Example: Minimal MCP Server Setup for CrewAI Integration

```json
// crewai.config.json
{
  "mcpServers": {
    "browserbase": {
      "command": "npx",
      "args": ["@browserbasehq/mcp-server-browserbase"],
      "env": {
        "BROWSERBASE_API_KEY": "your_api_key",
        "BROWSERBASE_PROJECT_ID": "your_project_id"
      }
    }
  }
}
```
- For remote: replace `"command"` and `"args"` with the remote URL as per your infrastructure[2].

---

## 10. Recommendations for Your Project

- **Automate MCP server startup** as part of your CLI or agent initialization scripts.
- **Parameterize session options** (user agent, proxy, viewport) in your `BrowserbaseManager` to leverage MCP server flexibility.
- **Monitor changelogs** for SDK and server updates, especially for anti-bot and session management improvements[4].
- **Write integration tests** that spin up a local MCP server with test credentials to validate end-to-end flows.

---

This setup will ensure robust, scalable, and maintainable integration of Browserbase MCP server into your scraping and agent automation pipeline, supporting both current and future requirements.


---

*Generated by Task Master Research Command*  
*Timestamp: 2025-07-31T16:32:01.620Z*
