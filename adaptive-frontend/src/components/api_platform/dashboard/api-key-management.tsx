import { useState } from "react";
import { useIsMobile } from "@/hooks/use-mobile";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { toast } from "sonner";
import { useUser } from "@clerk/clerk-react";

import * as z from "zod";

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
import {
  Drawer,
  DrawerClose,
  DrawerContent,
  DrawerDescription,
  DrawerFooter,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from "@/components/ui/drawer";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Copy, Key, Plus, RefreshCw, Trash2, AlertCircle } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

import { useCreateAPIKey } from "@/lib/hooks/api_keys/use-create-api-key";
import { useUpdateAPIKey } from "@/lib/hooks/api_keys/use-update-api-key";
import { useDeleteAPIKey } from "@/lib/hooks/api_keys/use-delete-api-key";
import { useGetAPIKeysByUserId } from "@/lib/hooks/api_keys/use-get-api-keys-user";

import {
  CreateAPIKeyRequest,
  UpdateAPIKeyRequest,
} from "@/services/api_keys/types";

// Form schema for creating a new API key
const createApiKeySchema = z.object({
  name: z.string().min(1, "Name is required"),
  expiration: z.enum(["never", "30days", "90days", "1year"]),
});

type CreateApiKeyFormValues = z.infer<typeof createApiKeySchema>;

export function ApiKeyManagement() {
  const { user } = useUser();
  const userId = user?.id;
  const isMobile = useIsMobile();

  // State for dialogs
  const [showNewKeyDialog, setShowNewKeyDialog] = useState(false);
  const [newKeyData, setNewKeyData] = useState<{
    name: string;
    key: string;
  } | null>(null);
  const [open, setOpen] = useState(false);

  // React Query hooks
  const { data: apiKeys, isLoading } = useGetAPIKeysByUserId(userId || "");
  const createAPIKeyMutation = useCreateAPIKey();
  const updateAPIKeyMutation = useUpdateAPIKey();
  const deleteAPIKeyMutation = useDeleteAPIKey();

  // Form setup
  const form = useForm<CreateApiKeyFormValues>({
    resolver: zodResolver(createApiKeySchema),
    defaultValues: {
      name: "",
      expiration: "never",
    },
  });

  // Create a new API key
  const onSubmit = (values: CreateApiKeyFormValues) => {
    if (!userId) return;

    // Calculate expiration date if needed
    let expiresAt: string | undefined;
    const now = new Date();

    switch (values.expiration) {
      case "30days":
        now.setDate(now.getDate() + 30);
        expiresAt = now.toISOString();
        break;
      case "90days":
        now.setDate(now.getDate() + 90);
        expiresAt = now.toISOString();
        break;
      case "1year":
        now.setFullYear(now.getFullYear() + 1);
        expiresAt = now.toISOString();
        break;
      default:
        expiresAt = undefined;
    }

    const apiKeyData: CreateAPIKeyRequest = {
      name: values.name,
      user_id: userId,
      status: "active",
      expires_at: expiresAt,
    };

    createAPIKeyMutation.mutate(apiKeyData, {
      onSuccess: (response) => {
        setOpen(false);
        form.reset();

        // Show the new key in a dialog
        setNewKeyData({
          name: response.api_key.name,
          key: response.full_api_key,
        });
        setShowNewKeyDialog(true);

        toast.success("API Key Created");
      },
      onError: (error: unknown) => {
        if (error instanceof Error) {
          toast.error(error.message);
        }
        toast.error("Failed to create API key.");
      },
    });
  };

  // Toggle API key status (active/inactive)
  const toggleKeyStatus = (id: string, currentStatus: string) => {
    const newStatus = currentStatus === "active" ? "inactive" : "active";

    updateAPIKeyMutation.mutate(
      {
        id,
        apiKeyData: {
          status: newStatus,
        } as UpdateAPIKeyRequest,
      },
      {
        onSuccess: () => {
          toast.success("API Key Updated");
        },
        onError: (error) => {
          toast.error(error.message);
        },
      },
    );
  };

  // Delete an API key
  const deleteKey = (id: string) => {
    if (!userId) return;

    deleteAPIKeyMutation.mutate(
      { id, userId },
      {
        onSuccess: () => {
          toast.success("API Key Deleted");
        },
        onError: (error) => {
          toast.error(error.message);
        },
      },
    );
  };

  // Copy API key to clipboard
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text).then(
      () => {
        toast.info("API key copied to clipboard.");
      },
      (err) => {
        toast.error(err.message);
      },
    );
  };

  // Format date for display
  const formatDate = (dateString: string | null | undefined) => {
    if (!dateString) return "-";
    return new Date(dateString).toLocaleDateString();
  };

  // Format API key for display (mask it)
  const formatApiKey = (keyPrefix: string) => {
    return `${keyPrefix}••••••••••••••••`;
  };

  // Create API Key Form component
  const CreateApiKeyForm = () => (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
        <FormField
          control={form.control}
          name="name"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Key Name</FormLabel>
              <FormControl>
                <Input placeholder="e.g., Production API Key" {...field} />
              </FormControl>
              <FormDescription>
                Give your API key a descriptive name to identify its purpose.
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="expiration"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Expiration</FormLabel>
              <Select onValueChange={field.onChange} defaultValue={field.value}>
                <FormControl>
                  <SelectTrigger>
                    <SelectValue placeholder="Select expiration" />
                  </SelectTrigger>
                </FormControl>
                <SelectContent>
                  <SelectItem value="never">Never</SelectItem>
                  <SelectItem value="30days">30 days</SelectItem>
                  <SelectItem value="90days">90 days</SelectItem>
                  <SelectItem value="1year">1 year</SelectItem>
                </SelectContent>
              </Select>
              <FormDescription>
                Choose when this API key should expire.
              </FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />
        <div className="flex justify-end gap-2">
          {isMobile ? (
            <DrawerClose asChild>
              <Button variant="outline">Cancel</Button>
            </DrawerClose>
          ) : (
            <Button
              type="button"
              variant="outline"
              onClick={() => setOpen(false)}
            >
              Cancel
            </Button>
          )}
          <Button type="submit" disabled={createAPIKeyMutation.isPending}>
            {createAPIKeyMutation.isPending ? "Creating..." : "Create Key"}
          </Button>
        </div>
      </form>
    </Form>
  );

  // Responsive create API key dialog/drawer
  const CreateApiKeyDialog = () => {
    if (isMobile) {
      return (
        <Drawer open={open} onOpenChange={setOpen}>
          <DrawerTrigger asChild>
            <Button>
              <Plus className="mr-2 h-4 w-4" />
              Create New API Key
            </Button>
          </DrawerTrigger>
          <DrawerContent>
            <DrawerHeader>
              <DrawerTitle>Create New API Key</DrawerTitle>
              <DrawerDescription>
                Create a new API key to access the Adaptive API. Keep your API
                keys secure.
              </DrawerDescription>
            </DrawerHeader>
            <div className="px-4 py-2">
              <CreateApiKeyForm />
            </div>
          </DrawerContent>
        </Drawer>
      );
    }

    return (
      <Dialog open={open} onOpenChange={setOpen}>
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
          <CreateApiKeyForm />
        </DialogContent>
      </Dialog>
    );
  };

  // New API Key Display Dialog/Drawer
  const NewApiKeyDisplay = () => {
    const content = (
      <>
        <div className="my-4">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Important</AlertTitle>
            <AlertDescription>
              For security reasons, we don't store the full API key. Make sure
              to copy it now.
            </AlertDescription>
          </Alert>
        </div>
        {newKeyData && (
          <div className="space-y-4">
            <div>
              <Label>Key Name</Label>
              <div className="mt-1 font-medium">{newKeyData.name}</div>
            </div>
            <div>
              <Label>API Key</Label>
              <div className="mt-1 flex items-center justify-between rounded-md border p-2 overflow-x-auto">
                <code className="font-mono text-sm">{newKeyData.key}</code>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => copyToClipboard(newKeyData.key)}
                  className="ml-2 shrink-0"
                >
                  <Copy className="h-4 w-4 mr-2" />
                  Copy
                </Button>
              </div>
            </div>
          </div>
        )}
      </>
    );

    if (isMobile) {
      return (
        <Drawer open={showNewKeyDialog} onOpenChange={setShowNewKeyDialog}>
          <DrawerContent>
            <DrawerHeader>
              <DrawerTitle>Your New API Key</DrawerTitle>
              <DrawerDescription>
                This is your new API key. Please copy it now as you won't be
                able to see it again.
              </DrawerDescription>
            </DrawerHeader>
            <div className="px-4 py-2">{content}</div>
            <DrawerFooter>
              <Button onClick={() => setShowNewKeyDialog(false)}>
                I've Copied My Key
              </Button>
            </DrawerFooter>
          </DrawerContent>
        </Drawer>
      );
    }

    return (
      <Dialog open={showNewKeyDialog} onOpenChange={setShowNewKeyDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Your New API Key</DialogTitle>
            <DialogDescription>
              This is your new API key. Please copy it now as you won't be able
              to see it again.
            </DialogDescription>
          </DialogHeader>
          {content}
          <DialogFooter>
            <Button onClick={() => setShowNewKeyDialog(false)}>
              I've Copied My Key
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">API Key Management</h2>
        <CreateApiKeyDialog />
      </div>

      {/* New API Key Dialog/Drawer */}
      <NewApiKeyDisplay />

      <Card>
        <CardHeader>
          <CardTitle>Your API Keys</CardTitle>
          <CardDescription>
            Manage your API keys for accessing the Adaptive API. Keep your keys
            secure and never share them publicly.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            // Loading skeleton
            <div className="space-y-4">
              {[1, 2, 3].map((i) => (
                <div key={i} className="flex items-center space-x-4">
                  <Skeleton className="h-12 w-[250px]" />
                  <Skeleton className="h-12 w-[200px]" />
                  <Skeleton className="h-12 w-[100px]" />
                  <Skeleton className="h-12 w-[100px]" />
                  <Skeleton className="h-12 w-[80px]" />
                  <Skeleton className="h-12 w-[80px]" />
                </div>
              ))}
            </div>
          ) : apiKeys && apiKeys.length > 0 ? (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>API Key</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead className="hidden md:table-cell">
                      Last Used
                    </TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {apiKeys.map((key) => (
                    <TableRow key={key.id}>
                      <TableCell className="font-medium">{key.name}</TableCell>
                      <TableCell>
                        <div className="flex items-center space-x-2">
                          <Key className="h-4 w-4 text-muted-foreground" />
                          <span className="truncate max-w-[120px] md:max-w-[200px]">
                            {formatApiKey(key.key_prefix)}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell className="whitespace-nowrap">
                        {formatDate(key.created_at)}
                      </TableCell>
                      <TableCell className="hidden md:table-cell whitespace-nowrap">
                        {formatDate(key.last_used_at)}
                      </TableCell>
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
                            onClick={() => toggleKeyStatus(key.id, key.status)}
                            title={
                              key.status === "active"
                                ? "Deactivate"
                                : "Activate"
                            }
                            disabled={updateAPIKeyMutation.isPending}
                          >
                            <RefreshCw className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => deleteKey(key.id)}
                            title="Delete"
                            disabled={deleteAPIKeyMutation.isPending}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-muted-foreground">
                You don't have any API keys yet. Create one to get started.
              </p>
              <Button
                className="mt-4"
                variant="outline"
                onClick={() => setOpen(true)}
              >
                <Plus className="mr-2 h-4 w-4" />
                Create API Key
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
