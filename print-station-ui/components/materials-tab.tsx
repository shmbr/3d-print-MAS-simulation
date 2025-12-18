"use client"

import type React from "react"

import { useState, useMemo } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Textarea } from "@/components/ui/textarea"
import {
  Plus,
  Search,
  MoreHorizontal,
  Edit,
  Trash2,
  AlertTriangle,
  ArrowUpDown,
  Archive,
  MessageSquare,
} from "lucide-react"

export default function MaterialsTab() {
  const [materials, setMaterials] = useState([
    {
      id: 1,
      name: "PLA Білий",
      type: "PLA",
      color: "#ffffff",
      weight: 850,
      totalWeight: 1000,
      cost: 450,
      supplier: "3D Пластик Україна",
      purchaseDate: "2024-01-15",
      status: "active",
      comments: "Відмінна якість, легко друкується",
      printedModels: ["Кріплення для телефону", "Органайзер для інструментів"],
    },
    {
      id: 2,
      name: "PLA Червоний",
      type: "PLA",
      color: "#dc2626",
      weight: 50,
      totalWeight: 1000,
      cost: 450,
      supplier: "3D Пластик Україна",
      purchaseDate: "2024-01-10",
      status: "active",
      comments: "Яскравий колір, але швидко закінчується",
      printedModels: ["Мініатюра дракона", "Іграшковий робот"],
    },
    {
      id: 3,
      name: "ABS Чорний",
      type: "ABS",
      color: "#000000",
      weight: 750,
      totalWeight: 1000,
      cost: 520,
      supplier: "PrintBox",
      purchaseDate: "2024-01-20",
      status: "active",
      comments: "Міцний, підходить для технічних деталей",
      printedModels: ["Шестерня для принтера"],
    },
    {
      id: 4,
      name: "PETG Прозорий",
      type: "PETG",
      color: "#f3f4f6",
      weight: 300,
      totalWeight: 1000,
      cost: 680,
      supplier: "3D Пластик Україна",
      purchaseDate: "2024-01-25",
      status: "active",
      comments: "Красивий прозорий ефект",
      printedModels: ["Ваза спіральна"],
    },
    {
      id: 5,
      name: "PLA Старий синій",
      type: "PLA",
      color: "#2563eb",
      weight: 0,
      totalWeight: 1000,
      cost: 400,
      supplier: "PrintBox",
      purchaseDate: "2023-12-01",
      status: "archived",
      comments: "Закінчився, якість була середня",
      printedModels: ["Тестові моделі", "Прототипи"],
    },
  ])

  const [searchTerm, setSearchTerm] = useState("")
  const [typeFilter, setTypeFilter] = useState("all")
  const [statusFilter, setStatusFilter] = useState("active")
  const [sortField, setSortField] = useState("name")
  const [sortDirection, setSortDirection] = useState("asc")
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false)
  const [selectedMaterial, setSelectedMaterial] = useState(null)
  const [isDetailsDialogOpen, setIsDetailsDialogOpen] = useState(false)

  const [newMaterial, setNewMaterial] = useState({
    name: "",
    type: "PLA",
    color: "#ffffff",
    weight: "",
    cost: "",
    supplier: "",
    comments: "",
  })

  const addMaterial = () => {
    if (newMaterial.name && newMaterial.weight && newMaterial.cost) {
      setMaterials([
        ...materials,
        {
          id: Date.now(),
          ...newMaterial,
          weight: Number.parseInt(newMaterial.weight),
          totalWeight: Number.parseInt(newMaterial.weight),
          cost: Number.parseInt(newMaterial.cost),
          purchaseDate: new Date().toISOString().split("T")[0],
          status: "active",
          printedModels: [],
        },
      ])
      setNewMaterial({
        name: "",
        type: "PLA",
        color: "#ffffff",
        weight: "",
        cost: "",
        supplier: "",
        comments: "",
      })
      setIsAddDialogOpen(false)
    }
  }

  const deleteMaterial = (id: number) => {
    setMaterials(materials.filter((material) => material.id !== id))
  }

  const archiveMaterial = (id: number) => {
    setMaterials(materials.map((material) => (material.id === id ? { ...material, status: "archived" } : material)))
  }

  const getStatusColor = (percentage: number) => {
    if (percentage < 10) return "text-red-500"
    if (percentage < 30) return "text-yellow-500"
    return "text-green-500"
  }

  const getStatusBadge = (percentage: number, status: string) => {
    if (status === "archived") return <Badge variant="secondary">Архів</Badge>
    if (percentage < 10) return <Badge variant="destructive">Критично</Badge>
    if (percentage < 30) return <Badge variant="secondary">Мало</Badge>
    return <Badge variant="default">Достатньо</Badge>
  }

  const getStatus = (weight: number, totalWeight: number) => {
    const percentage = (weight / totalWeight) * 100
    if (percentage < 10) return "critical"
    if (percentage < 30) return "low"
    return "good"
  }

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc")
    } else {
      setSortField(field)
      setSortDirection("asc")
    }
  }

  const filteredAndSortedMaterials = useMemo(() => {
    const filtered = materials.filter((material) => {
      const matchesSearch =
        material.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        material.supplier.toLowerCase().includes(searchTerm.toLowerCase()) ||
        material.comments.toLowerCase().includes(searchTerm.toLowerCase())
      const matchesType = typeFilter === "all" || material.type === typeFilter
      const matchesStatus = statusFilter === "all" || material.status === statusFilter

      return matchesSearch && matchesType && matchesStatus
    })

    filtered.sort((a, b) => {
      let aValue, bValue

      switch (sortField) {
        case "name":
          aValue = a.name
          bValue = b.name
          break
        case "type":
          aValue = a.type
          bValue = b.type
          break
        case "weight":
          aValue = a.weight
          bValue = b.weight
          break
        case "percentage":
          aValue = (a.weight / a.totalWeight) * 100
          bValue = (b.weight / b.totalWeight) * 100
          break
        case "cost":
          aValue = a.cost
          bValue = b.cost
          break
        case "purchaseDate":
          aValue = new Date(a.purchaseDate)
          bValue = new Date(b.purchaseDate)
          break
        default:
          aValue = a.name
          bValue = b.name
      }

      if (typeof aValue === "string") {
        return sortDirection === "asc" ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue)
      } else {
        return sortDirection === "asc" ? aValue - bValue : bValue - aValue
      }
    })

    return filtered
  }, [materials, searchTerm, typeFilter, statusFilter, sortField, sortDirection])

  const SortableHeader = ({ field, children }: { field: string; children: React.ReactNode }) => (
    <TableHead className="cursor-pointer hover:bg-muted/50" onClick={() => handleSort(field)}>
      <div className="flex items-center gap-2">
        {children}
        <ArrowUpDown className="h-4 w-4" />
      </div>
    </TableHead>
  )

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">Матеріали</h2>
          <p className="text-muted-foreground">Управління запасами пластику</p>
        </div>
        <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Додати матеріал
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Додати новий матеріал</DialogTitle>
              <DialogDescription>Введіть інформацію про новий пластик</DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="name">Назва</Label>
                <Input
                  id="name"
                  value={newMaterial.name}
                  onChange={(e) => setNewMaterial({ ...newMaterial, name: e.target.value })}
                  placeholder="PLA Синій"
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="grid gap-2">
                  <Label htmlFor="type">Тип</Label>
                  <Select
                    value={newMaterial.type}
                    onValueChange={(value) => setNewMaterial({ ...newMaterial, type: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="PLA">PLA</SelectItem>
                      <SelectItem value="ABS">ABS</SelectItem>
                      <SelectItem value="PETG">PETG</SelectItem>
                      <SelectItem value="TPU">TPU</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="color">Колір</Label>
                  <Input
                    id="color"
                    type="color"
                    value={newMaterial.color}
                    onChange={(e) => setNewMaterial({ ...newMaterial, color: e.target.value })}
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="grid gap-2">
                  <Label htmlFor="weight">Вага (г)</Label>
                  <Input
                    id="weight"
                    type="number"
                    value={newMaterial.weight}
                    onChange={(e) => setNewMaterial({ ...newMaterial, weight: e.target.value })}
                    placeholder="1000"
                  />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="cost">Ціна (₴)</Label>
                  <Input
                    id="cost"
                    type="number"
                    value={newMaterial.cost}
                    onChange={(e) => setNewMaterial({ ...newMaterial, cost: e.target.value })}
                    placeholder="450"
                  />
                </div>
              </div>
              <div className="grid gap-2">
                <Label htmlFor="supplier">Постачальник</Label>
                <Input
                  id="supplier"
                  value={newMaterial.supplier}
                  onChange={(e) => setNewMaterial({ ...newMaterial, supplier: e.target.value })}
                  placeholder="3D Пластик Україна"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="comments">Коментарі</Label>
                <Textarea
                  id="comments"
                  value={newMaterial.comments}
                  onChange={(e) => setNewMaterial({ ...newMaterial, comments: e.target.value })}
                  placeholder="Особливості, якість, рекомендації..."
                />
              </div>
            </div>
            <DialogFooter>
              <Button onClick={addMaterial}>Додати</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Фільтри та пошук</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="space-y-2">
              <Label htmlFor="search">Пошук</Label>
              <div className="relative">
                <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  id="search"
                  placeholder="Назва, постачальник, коментарі..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-8"
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="typeFilter">Тип пластику</Label>
              <Select value={typeFilter} onValueChange={setTypeFilter}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Всі типи</SelectItem>
                  <SelectItem value="PLA">PLA</SelectItem>
                  <SelectItem value="ABS">ABS</SelectItem>
                  <SelectItem value="PETG">PETG</SelectItem>
                  <SelectItem value="TPU">TPU</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="statusFilter">Статус</Label>
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Всі статуси</SelectItem>
                  <SelectItem value="active">Активні</SelectItem>
                  <SelectItem value="archived">Архівні</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Результати</Label>
              <div className="text-sm text-muted-foreground pt-2">
                Знайдено: {filteredAndSortedMaterials.length} з {materials.length}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Materials Table */}
      <Card>
        <CardHeader>
          <CardTitle>Список матеріалів</CardTitle>
          <CardDescription>Клікніть на заголовок колонки для сортування</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <SortableHeader field="name">Назва</SortableHeader>
                <SortableHeader field="type">Тип</SortableHeader>
                <TableHead>Колір</TableHead>
                <SortableHeader field="weight">Залишок</SortableHeader>
                <SortableHeader field="percentage">Статус</SortableHeader>
                <SortableHeader field="cost">Ціна</SortableHeader>
                <TableHead>Постачальник</TableHead>
                <TableHead>Моделі</TableHead>
                <TableHead>Коментарі</TableHead>
                <TableHead className="w-[70px]">Дії</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredAndSortedMaterials.map((material) => {
                const percentage = Math.round((material.weight / material.totalWeight) * 100)
                return (
                  <TableRow key={material.id} className={material.status === "archived" ? "opacity-60" : ""}>
                    <TableCell className="font-medium">{material.name}</TableCell>
                    <TableCell>
                      <Badge variant="outline">{material.type}</Badge>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <div className="w-4 h-4 rounded-full border" style={{ backgroundColor: material.color }} />
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="space-y-1">
                        <div className="text-sm">
                          {material.weight}г / {material.totalWeight}г
                        </div>
                        {material.status === "active" && <Progress value={percentage} className="h-1 w-20" />}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        {getStatusBadge(percentage, material.status)}
                        {material.status === "active" && percentage < 30 && (
                          <AlertTriangle className="h-4 w-4 text-yellow-500" />
                        )}
                      </div>
                    </TableCell>
                    <TableCell>₴{material.cost}</TableCell>
                    <TableCell className="text-sm">{material.supplier}</TableCell>
                    <TableCell className="text-sm">
                      <div className="max-w-32 truncate">
                        {material.printedModels.length > 0 ? (
                          <span title={material.printedModels.join(", ")}>{material.printedModels.length} моделей</span>
                        ) : (
                          "—"
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="text-sm">
                      <div className="max-w-32 truncate">
                        {material.comments && (
                          <div className="flex items-center gap-1">
                            <MessageSquare className="h-3 w-3" />
                            <span title={material.comments}>Є коментар</span>
                          </div>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" className="h-8 w-8 p-0">
                            <MoreHorizontal className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuLabel>Дії</DropdownMenuLabel>
                          <DropdownMenuItem
                            onClick={() => {
                              setSelectedMaterial(material)
                              setIsDetailsDialogOpen(true)
                            }}
                          >
                            <Edit className="mr-2 h-4 w-4" />
                            Деталі
                          </DropdownMenuItem>
                          {material.status === "active" && (
                            <DropdownMenuItem onClick={() => archiveMaterial(material.id)}>
                              <Archive className="mr-2 h-4 w-4" />В архів
                            </DropdownMenuItem>
                          )}
                          <DropdownMenuSeparator />
                          <DropdownMenuItem className="text-red-600" onClick={() => deleteMaterial(material.id)}>
                            <Trash2 className="mr-2 h-4 w-4" />
                            Видалити
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>

          {filteredAndSortedMaterials.length === 0 && (
            <div className="text-center py-8 text-muted-foreground">
              Матеріали не знайдені. Спробуйте змінити фільтри.
            </div>
          )}
        </CardContent>
      </Card>

      {/* Material Details Dialog */}
      <Dialog open={isDetailsDialogOpen} onOpenChange={setIsDetailsDialogOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Деталі матеріалу: {selectedMaterial?.name}</DialogTitle>
          </DialogHeader>
          {selectedMaterial && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>Тип</Label>
                  <div className="text-sm">{selectedMaterial.type}</div>
                </div>
                <div>
                  <Label>Постачальник</Label>
                  <div className="text-sm">{selectedMaterial.supplier}</div>
                </div>
                <div>
                  <Label>Ціна</Label>
                  <div className="text-sm">₴{selectedMaterial.cost}</div>
                </div>
                <div>
                  <Label>Дата покупки</Label>
                  <div className="text-sm">{selectedMaterial.purchaseDate}</div>
                </div>
              </div>

              <div>
                <Label>Коментарі</Label>
                <div className="text-sm p-2 bg-muted rounded-md">{selectedMaterial.comments || "Немає коментарів"}</div>
              </div>

              <div>
                <Label>Надруковані моделі ({selectedMaterial.printedModels.length})</Label>
                <div className="space-y-1 max-h-32 overflow-y-auto">
                  {selectedMaterial.printedModels.length > 0 ? (
                    selectedMaterial.printedModels.map((model, index) => (
                      <div key={index} className="text-sm p-2 bg-muted rounded-md">
                        {model}
                      </div>
                    ))
                  ) : (
                    <div className="text-sm text-muted-foreground">Ще не використовувався</div>
                  )}
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}
