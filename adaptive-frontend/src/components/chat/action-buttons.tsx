import { 
  Eye, 
  FileText, 
  GraduationCap, 
  ImageIcon, 
  MoreHorizontal 
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface ActionButtonsProps {
  visible: boolean;
}

export function ActionButtons({ visible }: ActionButtonsProps) {
  if (!visible) return null;

  return (
    <div className="flex flex-wrap justify-center gap-2 pb-2 duration-200 animate-in fade-in slide-in-from-bottom-2">
      <Button variant="secondary" size="sm" className="gap-2 rounded-full">
        <ImageIcon className="w-4 h-4" />
        <span>Create image</span>
      </Button>
      <Button variant="secondary" size="sm" className="gap-2 rounded-full">
        <FileText className="w-4 h-4" />
        <span>Summarize text</span>
      </Button>
      <Button variant="secondary" size="sm" className="gap-2 rounded-full">
        <Eye className="w-4 h-4" />
        <span>Analyze images</span>
      </Button>
      <Button variant="secondary" size="sm" className="gap-2 rounded-full">
        <GraduationCap className="w-4 h-4" />
        <span>Get advice</span>
      </Button>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="secondary" size="sm" className="rounded-full">
            <MoreHorizontal className="w-4 h-4" />
            <span className="sr-only">More options</span>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem>Code generation</DropdownMenuItem>
          <DropdownMenuItem>Data analysis</DropdownMenuItem>
          <DropdownMenuItem>Translation</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}