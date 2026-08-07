"use client";
import { useSample } from "@/app/contexts/SampleContext";
import { getSignalNames, shallowEqual } from "@/app/utils";
import { Profile2DViewParams } from "@/types";
import { ComboBox, Flex, Item, Switch } from "@adobe/react-spectrum";
import { useEffect, useMemo, useState } from "react";

export function Profile2DViewParamsWidget() {
  const { sample, viewParams, setViewParams } = useSample();
  // Seed from any already-restored value in context instead of starting blank.
  const initialParams =
    viewParams.name === "profile_2d"
      ? (viewParams as Profile2DViewParams)
      : null;
  const [selectedSignal, setSelectedSignal] = useState<string | null>(
    initialParams?.signal_name ?? null,
  );
  const [logScale, setLogScale] = useState<boolean>(
    initialParams?.log_scale ?? false,
  );

  const signalNames = useMemo(() => getSignalNames(sample), [sample]);

  useEffect(() => {
    if (signalNames.length > 0 && !selectedSignal) {
      setSelectedSignal(signalNames[0]);
    }
  }, [signalNames, selectedSignal]);

  useEffect(() => {
    if (!selectedSignal) return;

    setViewParams((prevParams) => {
      const nextParams: Profile2DViewParams = {
        ...(prevParams as Profile2DViewParams),
        name: "profile_2d",
        signal_name: selectedSignal,
        log_scale: logScale,
      };

      // Only update if the params actually changed 
      return shallowEqual(prevParams, nextParams) ? prevParams : nextParams;
    });
  }, [selectedSignal, logScale, setViewParams]);

  return (
    <Flex direction="column" alignItems="start" gap="size-200">
      <ComboBox
        label="Select Signal"
        selectedKey={selectedSignal}
        onSelectionChange={(key) => setSelectedSignal(key as string)}
      >
        {signalNames.map((signal) => (
          <Item key={signal}>{signal}</Item>
        ))}
      </ComboBox>
      <Switch isSelected={logScale} onChange={setLogScale}>
        Log Scale
      </Switch>
    </Flex>
  );
}
