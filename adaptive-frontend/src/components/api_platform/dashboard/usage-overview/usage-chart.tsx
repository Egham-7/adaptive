import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const data = [
  { date: "Jan 01", requests: 1200, tokens: 240000 },
  { date: "Jan 02", requests: 1300, tokens: 260000 },
  { date: "Jan 03", requests: 1400, tokens: 280000 },
  { date: "Jan 04", requests: 1500, tokens: 300000 },
  { date: "Jan 05", requests: 1600, tokens: 320000 },
  { date: "Jan 06", requests: 1700, tokens: 340000 },
  { date: "Jan 07", requests: 1800, tokens: 360000 },
  { date: "Jan 08", requests: 1900, tokens: 380000 },
  { date: "Jan 09", requests: 2000, tokens: 400000 },
  { date: "Jan 10", requests: 2100, tokens: 420000 },
  { date: "Jan 11", requests: 2200, tokens: 440000 },
  { date: "Jan 12", requests: 2300, tokens: 460000 },
  { date: "Jan 13", requests: 2400, tokens: 480000 },
  { date: "Jan 14", requests: 2500, tokens: 500000 },
];

export function UsageChart() {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart
        data={data}
        margin={{
          top: 5,
          right: 10,
          left: 10,
          bottom: 0,
        }}
      >
        <XAxis
          dataKey="date"
          tickLine={false}
          axisLine={false}
          tickMargin={10}
          stroke="#888888"
          fontSize={12}
        />
        <YAxis
          yAxisId="left"
          tickLine={false}
          axisLine={false}
          tickMargin={10}
          stroke="#888888"
          fontSize={12}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          tickLine={false}
          axisLine={false}
          tickMargin={10}
          stroke="#888888"
          fontSize={12}
        />
        <Tooltip />
        <Line
          yAxisId="left"
          type="monotone"
          dataKey="requests"
          stroke="#8884d8"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 6 }}
        />
        <Line
          yAxisId="right"
          type="monotone"
          dataKey="tokens"
          stroke="#82ca9d"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 6 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
