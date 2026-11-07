import { expect, test, type Locator, type Page } from "@playwright/test";
import {
  capabilities,
  effective,
  fakeApi,
  globalConfiguration,
} from "../../sandbox/fixtures";

/** Whether the Desktop host must keep running this suite as well.
 *
 * Model limits and fallbacks are saved through each host's settings transport.
 */
export const desktopHostSensitive = true;

/** The add control resets to its placeholder after each pick. */
async function addFallback(trigger: Locator, name: string) {
  await trigger.click();
  const options = trigger.page().getByRole("listbox");
  await options.getByRole("option", { name, exact: true }).click();
  await expect(options).toBeHidden();
}

const llm = {
  ...effective.llm,
  preset: null,
  models: {
    default_model: { provider: "default", model: "deepseek-flash" },
    editor: { provider: "default", model: "editor-model" },
    backup: { provider: "default", model: "backup-model" },
    plain: { provider: "default", model: "Index-Translate-35B-A3B" },
  },
};

async function savedSettings(page: Page) {
  let saved = {
    ...globalConfiguration,
    effective: { ...effective, llm },
    yaml: JSON.stringify({ ...effective, llm }),
  };
  await fakeApi(page, {
    "/capabilities": {
      ...capabilities,
      operations: [
        { id: "translation.body", tier: "strong", description: "Translate paragraphs" },
        {
          id: "review.verify",
          tier: "strong",
          description: "Verify issues",
          resumable_conversation: true,
        },
      ],
    },
  });
  await page.route("**/api/settings", async (route) => {
    if (route.request().method() === "PUT") {
      const input = route.request().postDataJSON();
      saved = { ...input, effective: JSON.parse(input.yaml), revision: saved.revision + 1 };
    }
    await route.fulfill({ json: saved });
  });
  return () => saved.effective.llm as Record<string, any>;
}

export function registerTests() {
  test("model limits, batch output and step fallbacks save and reload", async ({ page }, testInfo) => {
    const saved = await savedSettings(page);
    await page.goto("/settings/providers");
    const window = page.locator("#context_window-plain");
    await window.fill("0");
    await expect(page.locator("#context_window-plain-error")).toHaveText(
      "Enter a positive whole number, or leave blank if unknown.",
    );
    await window.fill("24576");
    await expect(page.locator("#context_window-plain-error")).toHaveCount(0);
    await page.locator("#max_output_tokens-plain").fill("4096");
    await expect(page.locator("#max_output_tokens-editor")).toHaveAttribute("placeholder", "Unknown");
    await page.locator("#batch_output-plain").uncheck();
    await expect(page.getByText(/Turn off to request each paragraph separately/).first()).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath("model-limits.png"), fullPage: true });

    await page.getByRole("link", { name: "Default models", exact: true }).click();
    const add = page.getByLabel("Add fallback model for Translate chapters in batches", {
      exact: true,
    });
    await add.click();
    await expect(page.getByRole("option", { name: /^default_model/ })).toHaveCount(0);
    await page.keyboard.press("Escape");
    await addFallback(add, "editor · default / editor-model");
    await expect(add).toHaveText("Add fallback model");
    await addFallback(add, "backup · default / backup-model");
    const list = page.getByRole("list", {
      name: "Fallback models for Translate chapters in batches",
    });
    await expect(list.getByRole("listitem")).toHaveText(["editor", "backup"]);
    await page.getByRole("button", { name: "Move backup up", exact: true }).click();
    await expect(list.getByRole("listitem")).toHaveText(["backup", "editor"]);
    await expect(
      page.getByText("Not available: this step resumes one model conversation.", { exact: true }),
    ).toBeVisible();
    await expect(page.getByLabel(/Add fallback model for .*Verify/)).toHaveCount(0);
    await page.screenshot({ path: testInfo.outputPath("route-fallbacks.png"), fullPage: true });

    await page.getByRole("button", { name: "Save configuration", exact: true }).click();
    await expect(page.getByText("Global settings saved", { exact: true })).toBeVisible();
    expect(saved().models.plain).toEqual({
      provider: "default",
      model: "Index-Translate-35B-A3B",
      context_window: 24576,
      max_output_tokens: 4096,
      batch_output: false,
    });
    expect(saved().models.editor).toEqual({ provider: "default", model: "editor-model" });
    expect(saved().routes["translation.body"]).toEqual({
      tier: "strong",
      fallbacks: ["backup", "editor"],
    });

    await page.reload();
    await expect(list.getByRole("listitem")).toHaveText(["backup", "editor"]);
    await page.getByRole("button", { name: "Remove fallback backup", exact: true }).click();
    await expect(list.getByRole("listitem")).toHaveText(["editor"]);
    await page.getByRole("link", { name: "Models & connections", exact: true }).click();
    await expect(page.locator("#context_window-plain")).toHaveValue("24576");
    await expect(page.locator("#max_output_tokens-plain")).toHaveValue("4096");
    await expect(page.locator("#batch_output-plain")).not.toBeChecked();
    await expect(page.locator("#batch_output-editor")).toBeChecked();
  });

  test("model limit and fallback labels are translated", async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem("wenyi.locale", "zh-CN"));
    await savedSettings(page);
    await page.goto("/settings/providers");
    await expect(page.getByLabel("上下文窗口（token）", { exact: true }).first()).toBeVisible();
    await expect(page.getByLabel("最大输出（token）", { exact: true }).first()).toBeVisible();
    await expect(page.getByText("关闭后每段单独请求，适合专用翻译模型；请求次数更多。").first()).toBeVisible();
    await page.getByRole("link", { name: "默认模型", exact: true }).click();
    await expect(page.getByText("不可用：此步骤会续接同一模型的会话。", { exact: true })).toBeVisible();
    await expect(
      page.getByRole("main").evaluate((node) => node.scrollWidth <= node.clientWidth),
    ).resolves.toBe(true);
  });
}
