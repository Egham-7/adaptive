import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Copy, Key, Plus, RefreshCw, Trash2 } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const apiKeys = [
  {
    id: "key_1",
    name: "Production API Key",
    key: "adpt_prod_••••••••••••••••",
    created: "2023-10-15",
    lastUsed: "2023-11-28",
    status: "active",
  },
  {
    id: "key_2",
    name: "Development API Key",
    key: "adpt_dev_•••••••••••••••••",
    created: "2023-11-01",
    lastUsed: "2023-11-27",
    status: "active",
  },
  {
    id: "key_3",
    name: "Testing API Key",
    key: "adpt_test_••••••••••••••••",
    created: "2023-11-10",
    lastUsed: "2023-11-20",
    status: "inactive",
  },
];

export function ApiKeyManagement() {
  const [keys, setKeys] = useState(apiKeys);
  const [newKeyDialogOpen, setNewKeyDialogOpen] = useState(false);
  const [newKeyName, setNewKeyName] = useState("");

  const createNewKey = () => {
    // In a real app, this would call an API to create a new key
    const newKey = {
      id: `key_${keys.length + 1}`,
      name: newKeyName,
      key: `adpt_${Math.random().toString(36).substring(2, 10)}••••••••••••`,
      created: new Date().toISOString().split("T")[0],
      lastUsed: "-",
      status: "active",
    };

    setKeys([...keys, newKey]);
    setNewKeyDialogOpen(false);
    setNewKeyName("");
  };

  const toggleKeyStatus = (id: string) => {
    setKeys(
      keys.map((key) =>
        key.id === id
          ? { ...key, status: key.status === "active" ? "inactive" : "active" }
          : key,
      ),
    );
  };

  const deleteKey = (id: string) => {
    setKeys(keys.filter((key) => key.id !== id));
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">API Key Management</h2>
        <Dialog open={newKeyDialogOpen} onOpenChange={setNewKeyDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="mr-2 h-4 w-4" />
              Create New API Key
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create New API Key</DialogTitle>
              <DialogDescription>
                Create a new API key to access the Adaptive API. Keep your API
                keys secure.
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="name">Key Name</Label>
                <Input
                  id="name"
                  placeholder="e.g., Production API Key"
                  value={newKeyName}
                  onChange={(e) => setNewKeyName(e.target.value)}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="expiration">Expiration</Label>
                <Select defaultValue="never">
                  <SelectTrigger id="expiration">
                    <SelectValue placeholder="Select expiration" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="never">Never</SelectItem>
                    <SelectItem value="30days">30 days</SelectItem>
                    <SelectItem value="90days">90 days</SelectItem>
                    <SelectItem value="1year">1 year</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => setNewKeyDialogOpen(false)}
              >
                Cancel
              </Button>
              <Button onClick={createNewKey} disabled={!newKeyName}>
                Create Key
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Your API Keys</CardTitle>
          <CardDescription>
            Manage your API keys for accessing the Adaptive API. Keep your keys
            secure and never share them publicly.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>API Key</TableHead>
                <TableHead>Created</TableHead>
                <TableHead>Last Used</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {keys.map((key) => (
                <TableRow key={key.id}>
                  <TableCell className="font-medium">{key.name}</TableCell>
                  <TableCell>
                    <div className="flex items-center space-x-2">
                      <Key className="h-4 w-4 text-muted-foreground" />
                      <span>{key.key}</span>
                      <Button variant="ghost" size="icon" className="h-6 w-6">
                        <Copy className="h-4 w-4" />
                      </Button>
                    </div>
                  </TableCell>
                  <TableCell>{key.created}</TableCell>
                  <TableCell>{key.lastUsed}</TableCell>
                  <TableCell>
                    <Badge
                      variant={
                        key.status === "active" ? "default" : "secondary"
                      }
                    >
                      {key.status}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <div className="flex space-x-1">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => toggleKeyStatus(key.id)}
                        title={
                          key.status === "active" ? "Deactivate" : "Activate"
                        }
                      >
                        <RefreshCw className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => deleteKey(key.id)}
                        title="Delete"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
