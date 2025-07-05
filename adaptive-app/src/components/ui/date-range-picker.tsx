import { CalendarIcon, ChevronDownIcon } from "lucide-react";
import { format } from "date-fns";
import type { DateRange } from "react-day-picker";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useState } from "react";

interface DateRangePickerProps {
  dateRange: DateRange;
  onDateRangeChange: (range: DateRange) => void;
  presetRanges?: Array<{
    label: string;
    value: string;
    range: DateRange;
  }>;
  onPresetSelect?: (preset: string) => void;
  className?: string;
}

export function DateRangePicker({
  dateRange,
  onDateRangeChange,
  presetRanges,
  onPresetSelect,
  className,
}: DateRangePickerProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className={cn("flex items-center gap-2", className)}>
      {/* Preset Selector */}
      <Select onValueChange={onPresetSelect}>
        <SelectTrigger className="w-40">
          <SelectValue placeholder="Choose preset range" />
        </SelectTrigger>
        <SelectContent>
          {presetRanges?.map((preset) => (
            <SelectItem key={preset.value} value={preset.value}>
              {preset.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* Custom Date Range Picker */}
      <Popover open={isOpen} onOpenChange={setIsOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            className={cn(
              "w-80 justify-start text-left font-normal",
              !dateRange && "text-muted-foreground",
            )}
          >
            <CalendarIcon className="mr-2 h-4 w-4" />
            {dateRange?.from ? (
              dateRange.to ? (
                <>
                  {format(dateRange.from, "LLL dd, y")} -{" "}
                  {format(dateRange.to, "LLL dd, y")}
                </>
              ) : (
                format(dateRange.from, "LLL dd, y")
              )
            ) : (
              <span>Pick a date range</span>
            )}
            <ChevronDownIcon className="ml-auto h-4 w-4 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-auto p-0" align="start">
          <Calendar
            mode="range"
            defaultMonth={dateRange.from}
            selected={dateRange}
            onSelect={(range) => {
              if (range) {
                onDateRangeChange(range);
              }
            }}
            numberOfMonths={2}
            required={false}
          />
          <div className="p-3 border-t">
            <Button
              onClick={() => setIsOpen(false)}
              className="w-full"
              size="sm"
            >
              Apply Range
            </Button>
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
}
