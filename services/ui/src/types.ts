export type TimeSeriesData = Array<{
  time: number;
  value: number;
}>;

export type SpectrogramData = Array<{
  time: number;
  frequency: number;
  amplitude: number;
}>;

export type Category = {
  name: string;
  color: string;
};

export type Zone = {
  category: Category;

  x0: number;
  x1: number;
};

export type VSpan = {
  category: Category;
  x: number;
};
