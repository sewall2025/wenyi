import { ArrowDown, ArrowUp, X } from "lucide-react";
import { Button } from "../../components/ui/button";
import { Label } from "../../components/ui/form";
import { Select, SelectItem } from "../../components/ui/select";
import { useI18n } from "../../i18n";
import { operationLabel } from "../../i18n/labels";
import { SettingsGroup } from "./SettingsSection";

type Document = Record<string, unknown>;
const object = (value: unknown) => (value || {}) as Document;

export function ModelSelection({
  scope,
  llm,
  models,
  operations = [],
  disabled,
  onChange,
}: {
  scope: "global" | "project";
  llm: Document;
  models: Document;
  operations?: Document[];
  disabled: boolean;
  onChange: (llm: Document) => void;
}) {
  const { t } = useI18n();
  const tiers = object(llm.tiers);
  const routes = object(llm.routes);
  const modelOptions = Object.entries(models).map(([id, raw]) => {
    const model = object(raw);
    return (
      <SelectItem key={id} value={id}>
        {id} · {String(model.provider)} / {String(model.model)}
      </SelectItem>
    );
  });
  const tierNames = [
    ["strong", t("providerSettings.qualityTier")],
    ["cheap", t("providerSettings.economyTier")],
    ["fast", t("providerSettings.fastTier")],
  ];
  return (
    <fieldset disabled={disabled} className="min-w-0 space-y-4 disabled:opacity-60">
      <div className="grid grid-cols-3 gap-3">
        {tierNames.map(([id, label]) => (
          <div key={id} className="min-w-0">
            <Label htmlFor={`tier-${id}`}>{label}</Label>
            <Select
              id={`tier-${id}`}
              value={String(tiers[id] || "")}
              onValueChange={(value) =>
                onChange({
                  ...llm,
                  tiers: { ...tiers, [id]: value },
                })
              }
            >
              {modelOptions}
            </Select>
          </div>
        ))}
      </div>
      <SettingsGroup title={t("settings.operationModels")} collapsible={scope === "project"}>
        <div className="space-y-3">
          {operations.map((operation) => {
            const id = String(operation.id);
            const route = object(routes[id]);
            const value = route.model
              ? String(route.model)
              : `tier:${route.tier || operation.tier}`;
            const primary = route.model
              ? String(route.model)
              : String(tiers[String(route.tier || operation.tier || "strong")] || "");
            const fallbacks = Array.isArray(route.fallbacks)
              ? route.fallbacks.map(String)
              : [];
            // Resumable evidence conversations must stay on one model; the backend rejects them.
            const locked = Boolean(operation.resumable_conversation);
            const save = (selected: string, nextFallbacks: string[]) => {
              const next = { ...routes };
              if (selected === `tier:${operation.tier}` && nextFallbacks.length === 0)
                delete next[id];
              else
                next[id] = {
                  ...(selected.startsWith("tier:")
                    ? { tier: selected.slice(5) }
                    : { model: selected }),
                  fallbacks: nextFallbacks,
                };
              onChange({ ...llm, routes: next });
            };
            const move = (index: number, offset: number) => {
              const next = [...fallbacks];
              [next[index], next[index + offset]] = [next[index + offset], next[index]];
              save(value, next);
            };
            const candidates = Object.keys(models).filter(
              (model) => model !== primary && !fallbacks.includes(model),
            );
            const label = operationLabel(id, t);
            return (
              <div key={id} className="space-y-2">
                <div className="grid grid-cols-2 items-center gap-2">
                  <Label
                    htmlFor={`operation-${id}`}
                    className="text-sm [overflow-wrap:anywhere]"
                  >
                    {label}
                  </Label>
                  <Select
                    id={`operation-${id}`}
                    value={value}
                    onValueChange={(selected) => {
                      const nextPrimary = selected.startsWith("tier:")
                        ? String(tiers[selected.slice(5)] || "")
                        : selected;
                      save(
                        selected,
                        fallbacks.filter((model) => model !== nextPrimary),
                      );
                    }}
                  >
                    {tierNames.map(([tier, tierLabel]) => (
                      <SelectItem key={tier} value={`tier:${tier}`}>
                        {tierLabel}
                      </SelectItem>
                    ))}
                    {modelOptions}
                  </Select>
                </div>
                <div className="grid grid-cols-2 items-start gap-2">
                  <span className="text-xs text-muted-foreground">
                    {t("settings.fallbackModels")}
                  </span>
                  <div className="min-w-0 space-y-1">
                    {locked ? (
                      <p className="text-xs text-muted-foreground">
                        {t("settings.fallbacksUnavailable")}
                      </p>
                    ) : (
                      <>
                        <ol
                          aria-label={t("settings.fallbacksFor", { operation: label })}
                          className="space-y-1"
                        >
                          {fallbacks.map((model, index) => (
                            <li key={model} className="flex items-center gap-1 text-sm">
                              <span className="min-w-0 flex-1 truncate">{model}</span>
                              <Button
                                type="button"
                                variant="ghost"
                                size="icon"
                                disabled={index === 0}
                                aria-label={t("settings.moveFallbackUp", { model })}
                                onClick={() => move(index, -1)}
                              >
                                <ArrowUp className="h-4 w-4" aria-hidden="true" />
                              </Button>
                              <Button
                                type="button"
                                variant="ghost"
                                size="icon"
                                disabled={index === fallbacks.length - 1}
                                aria-label={t("settings.moveFallbackDown", { model })}
                                onClick={() => move(index, 1)}
                              >
                                <ArrowDown className="h-4 w-4" aria-hidden="true" />
                              </Button>
                              <Button
                                type="button"
                                variant="ghost"
                                size="icon"
                                aria-label={t("settings.removeFallback", { model })}
                                onClick={() =>
                                  save(value, fallbacks.filter((item) => item !== model))
                                }
                              >
                                <X className="h-4 w-4" aria-hidden="true" />
                              </Button>
                            </li>
                          ))}
                        </ol>
                        {candidates.length > 0 && (
                          <Select
                            aria-label={t("settings.addFallbackFor", { operation: label })}
                            value=""
                            placeholder={t("settings.addFallback")}
                            onValueChange={(model) => save(value, [...fallbacks, model])}
                          >
                            {candidates.map((model) => (
                              <SelectItem key={model} value={model}>
                                {model} · {String(object(models[model]).provider)} /{" "}
                                {String(object(models[model]).model)}
                              </SelectItem>
                            ))}
                          </Select>
                        )}
                      </>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </SettingsGroup>
    </fieldset>
  );
}
