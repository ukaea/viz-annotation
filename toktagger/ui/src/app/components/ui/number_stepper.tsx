import { ActionButton, Flex, NumberField } from "@adobe/react-spectrum";
import { useEffect, useState } from "react";

export default function NumberStepper({
  label,
  defaultValue,
  onChange,
}: {
  label: string;
  defaultValue: number;
  onChange: (value: number) => void;
}) {
  const [value, setValue] = useState(defaultValue);

  const incrementValue = (increment: number) => {
    setValue((prevValue) => {
      const newValue = prevValue + increment;
      if (newValue < 0) return 0;
      if (newValue > 99) return 99;
      return newValue;
    });
  };

  useEffect(() => {
    onChange(value);
  }, [value, onChange]);

  return (
    <Flex direction="column" gap="size-100" alignItems="start" width="100%">
      <NumberField
        label={label}
        width="100%"
        value={value}
        onChange={setValue}
        minValue={0}
        maxValue={99}
        hideStepper={true}
      />
      <Flex direction="row" gap="size-100">
        <ActionButton
          onPress={() => {
            incrementValue(-5);
          }}
        >
          -5
        </ActionButton>
        <ActionButton
          onPress={() => {
            incrementValue(-1);
          }}
        >
          -1
        </ActionButton>
        <ActionButton
          onPress={() => {
            incrementValue(1);
          }}
        >
          +1
        </ActionButton>
        <ActionButton
          onPress={() => {
            incrementValue(5);
          }}
        >
          +5
        </ActionButton>
      </Flex>
    </Flex>
  );
}
