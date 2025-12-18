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
import { Textarea } from "@/components/ui/textarea"
import {
  Plus,
  Search,
  MoreHorizontal,
  Edit,
  Trash2,
  Play,
  Download,
  Clock,
  Package,
  ArrowUpDown,
  Eye,
  Star,
  User,
  FileText,
} from "lucide-react"
import Image from "next/image"

export default function ModelsTab() {
  const [models, setModels] = useState([
    {
      id: 1,
      name: "Мініатюра дракона",
      category: "Декоративні",
      status: "printing",
      printTime: "4г 30хв",
      material: "PLA Червоний",
      additionalMaterials: ["Підтримки PLA Білий"],
      materialUsed: 45,
      timesPlinted: 3,
      lastPrint: "2024-01-28",
      notes: "Найкращі результати при 210°C",
      image: "/placeholder.svg?height=60&width=60&query=dragon miniature",
      fileSize: "2.4 MB",
      complexity: "Середня",
      customer: "",
      successRate: 100,
      files: [
        { name: "dragon_body.stl", size: "1.8 MB" },
        { name: "dragon_wings.stl", size: "0.6 MB" },
      ],
    },
    {
      id: 2,
      name: "Кріплення для телефону",
      category: "Технічні",
      status: "completed",
      printTime: "2г 15хв",
      material: "PLA Білий",
      additionalMaterials: [],
      materialUsed: 28,
      timesPlinted: 5,
      lastPrint: "2024-01-27",
      notes: "Потребує підтримок для нависаючих частин",
      image: "/placeholder.svg?height=60&width=60&query=phone holder mount",
      fileSize: "1.8 MB",
      complexity: "Проста",
      customer: "Іван Петренко",
      successRate: 80,
      files: [{ name: "phone_holder.stl", size: "1.8 MB" }],
    },
    {
      id: 3,
      name: "Шестерня для принтера",
      category: "Запчастини",
      status: "queued",
      printTime: "1г 45хв",
      material: "ABS Чорний",
      additionalMaterials: [],
      materialUsed: 22,
      timesPlinted: 1,
      lastPrint: "2024-01-20",
      notes: "Критично важлива деталь, тримати запас",
      image: "/placeholder.svg?height=60&width=60&query=printer gear part",
      fileSize: "0.9 MB",
      complexity: "Проста",
      customer: "",
      successRate: 100,
      files: [{ name: "gear_20teeth.stl", size: "0.9 MB" }],
    },
    {
      id: 4,
      name: "Ваза спіральна",
      category: "Декоративні",
      status: "ready",
      printTime: "6г 20хв",
      material: "PETG Прозорий",
      additionalMaterials: [],
      materialUsed: 85,
      timesPlinted: 2,
      lastPrint: "2024-01-25",
      notes: "Красиво виглядає з підсвіткою",
      image: "/placeholder.svg?height=60&width=60&query=spiral vase decorative",
      fileSize: "3.2 MB",
      complexity: "Складна",
      customer: "Марія Коваленко",
      successRate: 50,
      files: [{ name: "spiral_vase.stl", size: "3.2 MB" }],
    },
  ])

  const [searchTerm, setSearchTerm] = useState("")
  const [categoryFilter, setCategoryFilter] = useState("all")
  const [statusFilter, setStatusFilter] = useState("all")
  const [complexityFilter, setComplexityFilter] = useState("all")
  const [sortField, setSortField] = useState("name")
  const [sortDirection, setSortDirection] = useState("asc")
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false)
  const [selectedModel, setSelectedModel] = useState(null)
  const [isDetailsDialogOpen, setIsDetailsDialogOpen] = useState(false)

  const [newModel, setNewModel] = useState({
    name: "",
    category: "Декоративні",
    printTime: "",
    material: "",
    additionalMaterials: "",
    materialUsed: "",
    notes: "",
    complexity: "Середня",
    customer: "",
  })

  const addModel = () => {
    if (newModel.name && newModel.printTime && newModel.material) {
      const additionalMaterialsArray = newModel.additionalMaterials
        ? newModel.additionalMaterials
            .split(",")
            .map((m) => m.trim())
            .filter(Boolean)
        : []

      setModels([
        ...models,
        {
          id: Date.now(),
          ...newModel,
          additionalMaterials: additionalMaterialsArray,
          status: "ready",
          materialUsed: Number.parseInt(newModel.materialUsed) || 0,
          timesPlinted: 0,
          lastPrint: "",
          image: "/placeholder.svg?height=60&width=60&query=3d model",
          fileSize: "1.0 MB",
          successRate: 0,
          files: [{ name: "model.stl", size: "1.0 MB" }],
        },
      ])
      setNewModel({
        name: "",
        category: "Декоративні",
        printTime: "",
        material: "",
        additionalMaterials: "",
        materialUsed: "",
        notes: "",
        complexity: "Середня",
        customer: "",
      })
      setIsAddDialogOpen(false)
    }
  }

  const deleteModel = (id: number) => {
    setModels(models.filter((model) => model.id !== id))
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "printing":
        return <Badge className="bg-blue-500">Друкується</Badge>
      case "completed":
        return <Badge className="bg-green-500">Завершено</Badge>
      case "queued":
        return <Badge variant="secondary">В черзі</Badge>
      case "ready":
        return <Badge variant="outline">Готова</Badge>
      case "failed":
        return <Badge variant="destructive">Помилка</Badge>
      default:
        return <Badge variant="outline">{status}</Badge>
    }
  }

  const getCategoryColor = (category: string) => {
    switch (category) {
      case "Декоративні":
        return "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200"
      case "Технічні":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
      case "Запчастини":
        return "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200"
      case "Іграшки":
        return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200"
    }
  }

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case "Проста":
        return "text-green-600"
      case "Середня":
        return "text-yellow-600"
      case "Складна":
        return "text-red-600"
      default:
        return "text-gray-600"
    }
  }

  const getSuccessRateColor = (rate: number) => {
    if (rate >= 80) return "text-green-600"
    if (rate >= 60) return "text-yellow-600"
    return "text-red-600"
  }

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc")
    } else {
      setSortField(field)
      setSortDirection("asc")
    }
  }

  const filteredAndSortedModels = useMemo(() => {
    const filtered = models.filter((model) => {
      const matchesSearch =
        model.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        model.material.toLowerCase().includes(searchTerm.toLowerCase()) ||
        model.notes.toLowerCase().includes(searchTerm.toLowerCase()) ||
        model.customer.toLowerCase().includes(searchTerm.toLowerCase())
      const matchesCategory = categoryFilter === "all" || model.category === categoryFilter
      const matchesStatus = statusFilter === "all" || model.status === statusFilter
      const matchesComplexity = complexityFilter === "all" || model.complexity === complexityFilter

      return matchesSearch && matchesCategory && matchesStatus && matchesComplexity
    })

    filtered.sort((a, b) => {
      let aValue, bValue

      switch (sortField) {
        case "name":
          aValue = a.name
          bValue = b.name
          break
        case "category":
          aValue = a.category
          bValue = b.category
          break
        case "status":
          aValue = a.status
          bValue = b.status
          break
        case "printTime":
          aValue = a.printTime.includes("г")
            ? Number.parseInt(a.printTime.split("г")[0]) * 60 +
              Number.parseInt(a.printTime.split("г")[1]?.replace("хв", "").trim() || "0")
            : Number.parseInt(a.printTime.replace("хв", ""))
          bValue = b.printTime.includes("г")
            ? Number.parseInt(b.printTime.split("г")[0]) * 60 +
              Number.parseInt(b.printTime.split("г")[1]?.replace("хв", "").trim() || "0")
            : Number.parseInt(b.printTime.replace("хв", ""))
          break
        case "materialUsed":
          aValue = a.materialUsed
          bValue = b.materialUsed
          break
        case "timesPlinted":
          aValue = a.timesPlinted
          bValue = b.timesPlinted
          break
        case "successRate":
          aValue = a.successRate
          bValue = b.successRate
          break
        case "lastPrint":
          aValue = a.lastPrint ? new Date(a.lastPrint) : new Date(0)
          bValue = b.lastPrint ? new Date(b.lastPrint) : new Date(0)
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
  }, [models, searchTerm, categoryFilter, statusFilter, complexityFilter, sortField, sortDirection])

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
          <h2 className="text-2xl font-bold">Бібліотека моделей</h2>
          <p className="text-muted-foreground">Управління 3D моделями та історією друку</p>
        </div>
        <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Додати модель
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>Додати нову модель</DialogTitle>
              <DialogDescription>Введіть інформацію про 3D модель</DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="grid gap-2">
                  <Label htmlFor="modelName">Назва моделі</Label>
                  <Input
                    id="modelName"
                    value={newModel.name}
                    onChange={(e) => setNewModel({ ...newModel, name: e.target.value })}
                    placeholder="Назва моделі"
                  />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="customer">Замовник (опціонально)</Label>
                  <Input
                    id="customer"
                    value={newModel.customer}
                    onChange={(e) => setNewModel({ ...newModel, customer: e.target.value })}
                    placeholder="Ім'я замовника"
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="grid gap-2">
                  <Label htmlFor="category">Категорія</Label>
                  <Select
                    value={newModel.category}
                    onValueChange={(value) => setNewModel({ ...newModel, category: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Декоративні">Декоративні</SelectItem>
                      <SelectItem value="Технічні">Технічні</SelectItem>
                      <SelectItem value="Запчастини">Запчастини</SelectItem>
                      <SelectItem value="Іграшки">Іграшки</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="complexity">Складність</Label>
                  <Select
                    value={newModel.complexity}
                    onValueChange={(value) => setNewModel({ ...newModel, complexity: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Проста">Проста</SelectItem>
                      <SelectItem value="Середня">Середня</SelectItem>
                      <SelectItem value="Складна">Складна</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="grid gap-2">
                  <Label htmlFor="printTime">Час друку</Label>
                  <Input
                    id="printTime"
                    value={newModel.printTime}
                    onChange={(e) => setNewModel({ ...newModel, printTime: e.target.value })}
                    placeholder="2г 30хв"
                  />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="materialUsed">Пластик (г)</Label>
                  <Input
                    id="materialUsed"
                    type="number"
                    value={newModel.materialUsed}
                    onChange={(e) => setNewModel({ ...newModel, materialUsed: e.target.value })}
                    placeholder="50"
                  />
                </div>
              </div>
              <div className="grid gap-2">
                <Label htmlFor="material">Основний матеріал</Label>
                <Input
                  id="material"
                  value={newModel.material}
                  onChange={(e) => setNewModel({ ...newModel, material: e.target.value })}
                  placeholder="PLA Білий"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="additionalMaterials">Додаткові матеріали</Label>
                <Input
                  id="additionalMaterials"
                  value={newModel.additionalMaterials}
                  onChange={(e) => setNewModel({ ...newModel, additionalMaterials: e.target.value })}
                  placeholder="Підтримки PLA Білий, Розчинні підтримки"
                />
                <p className="text-xs text-muted-foreground">Розділіть комами</p>
              </div>
              <div className="grid gap-2">
                <Label htmlFor="notes">Нотатки</Label>
                <Textarea
                  id="notes"
                  value={newModel.notes}
                  onChange={(e) => setNewModel({ ...newModel, notes: e.target.value })}
                  placeholder="Особливості друку, налаштування..."
                />
              </div>
            </div>
            <DialogFooter>
              <Button onClick={addModel}>Додати</Button>
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
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            <div className="space-y-2">
              <Label htmlFor="search">Пошук</Label>
              <div className="relative">
                <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  id="search"
                  placeholder="Назва, матеріал, замовник..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-8"
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="categoryFilter">Категорія</Label>
              <Select value={categoryFilter} onValueChange={setCategoryFilter}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Всі категорії</SelectItem>
                  <SelectItem value="Декоративні">Декоративні</SelectItem>
                  <SelectItem value="Технічні">Технічні</SelectItem>
                  <SelectItem value="Запчастини">Запчастини</SelectItem>
                  <SelectItem value="Іграшки">Іграшки</SelectItem>
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
                  <SelectItem value="ready">Готова</SelectItem>
                  <SelectItem value="printing">Друкується</SelectItem>
                  <SelectItem value="completed">Завершено</SelectItem>
                  <SelectItem value="queued">В черзі</SelectItem>
                  <SelectItem value="failed">Помилка</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="complexityFilter">Складність</Label>
              <Select value={complexityFilter} onValueChange={setComplexityFilter}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Всі рівні</SelectItem>
                  <SelectItem value="Проста">Проста</SelectItem>
                  <SelectItem value="Середня">Середня</SelectItem>
                  <SelectItem value="Складна">Складна</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Результати</Label>
              <div className="text-sm text-muted-foreground pt-2">
                Знайдено: {filteredAndSortedModels.length} з {models.length}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Models Table */}
      <Card>
        <CardHeader>
          <CardTitle>Список моделей</CardTitle>
          <CardDescription>Клікніть на заголовок колонки для сортування</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[80px]">Превью</TableHead>
                <SortableHeader field="name">Назва</SortableHeader>
                <SortableHeader field="category">Категорія</SortableHeader>
                <SortableHeader field="status">Статус</SortableHeader>
                <SortableHeader field="printTime">Час друку</SortableHeader>
                <TableHead>Матеріал</TableHead>
                <SortableHeader field="materialUsed">Пластик (г)</SortableHeader>
                <SortableHeader field="timesPlinted">Друків</SortableHeader>
                <SortableHeader field="successRate">Успішність</SortableHeader>
                <TableHead>Замовник</TableHead>
                <TableHead>Файли</TableHead>
                <TableHead className="w-[70px]">Дії</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredAndSortedModels.map((model) => (
                <TableRow key={model.id}>
                  <TableCell>
                    <div className="relative w-12 h-12 rounded-md overflow-hidden">
                      <Image src={model.image || "/placeholder.svg"} alt={model.name} fill className="object-cover" />
                    </div>
                  </TableCell>
                  <TableCell className="font-medium">
                    <div>
                      <div>{model.name}</div>
                      <div className="text-xs text-muted-foreground">{model.fileSize}</div>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge className={getCategoryColor(model.category)}>{model.category}</Badge>
                  </TableCell>
                  <TableCell>{getStatusBadge(model.status)}</TableCell>
                  <TableCell>
                    <div className="flex items-center gap-1">
                      <Clock className="h-3 w-3 text-muted-foreground" />
                      {model.printTime}
                    </div>
                  </TableCell>
                  <TableCell className="text-sm">
                    <div>
                      <div>{model.material}</div>
                      {model.additionalMaterials.length > 0 && (
                        <div className="text-xs text-muted-foreground">
                          +{model.additionalMaterials.length} додаткових
                        </div>
                      )}
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-1">
                      <Package className="h-3 w-3 text-muted-foreground" />
                      {model.materialUsed}г
                    </div>
                  </TableCell>
                  <TableCell className="text-center">{model.timesPlinted}</TableCell>
                  <TableCell>
                    <div className="flex items-center gap-1">
                      <Star className="h-3 w-3 text-muted-foreground" />
                      <span className={getSuccessRateColor(model.successRate)}>{model.successRate}%</span>
                    </div>
                  </TableCell>
                  <TableCell className="text-sm">
                    {model.customer ? (
                      <div className="flex items-center gap-1">
                        <User className="h-3 w-3 text-muted-foreground" />
                        {model.customer}
                      </div>
                    ) : (
                      "—"
                    )}
                  </TableCell>
                  <TableCell className="text-sm">
                    <div className="flex items-center gap-1">
                      <FileText className="h-3 w-3 text-muted-foreground" />
                      {model.files.length} файлів
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
                        <DropdownMenuItem>
                          <Play className="mr-2 h-4 w-4" />
                          Друкувати
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={() => {
                            setSelectedModel(model)
                            setIsDetailsDialogOpen(true)
                          }}
                        >
                          <Eye className="mr-2 h-4 w-4" />
                          Деталі
                        </DropdownMenuItem>
                        <DropdownMenuItem>
                          <Download className="mr-2 h-4 w-4" />
                          Завантажити
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem>
                          <Edit className="mr-2 h-4 w-4" />
                          Редагувати
                        </DropdownMenuItem>
                        <DropdownMenuItem className="text-red-600" onClick={() => deleteModel(model.id)}>
                          <Trash2 className="mr-2 h-4 w-4" />
                          Видалити
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>

          {filteredAndSortedModels.length === 0 && (
            <div className="text-center py-8 text-muted-foreground">Моделі не знайдені. Спробуйте змінити фільтри.</div>
          )}
        </CardContent>
      </Card>

      {/* Model Details Dialog */}
      <Dialog open={isDetailsDialogOpen} onOpenChange={setIsDetailsDialogOpen}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle>Деталі моделі: {selectedModel?.name}</DialogTitle>
          </DialogHeader>
          {selectedModel && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="relative w-full h-48 rounded-lg overflow-hidden bg-muted">
                    <Image
                      src={selectedModel.image || "/placeholder.svg"}
                      alt={selectedModel.name}
                      fill
                      className="object-cover"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label>Категорія</Label>
                      <div className="text-sm">{selectedModel.category}</div>
                    </div>
                    <div>
                      <Label>Складність</Label>
                      <div className={`text-sm ${getComplexityColor(selectedModel.complexity)}`}>
                        {selectedModel.complexity}
                      </div>
                    </div>
                    <div>
                      <Label>Час друку</Label>
                      <div className="text-sm">{selectedModel.printTime}</div>
                    </div>
                    <div>
                      <Label>Витрати пластику</Label>
                      <div className="text-sm">{selectedModel.materialUsed}г</div>
                    </div>
                    <div>
                      <Label>Кількість друків</Label>
                      <div className="text-sm">{selectedModel.timesPlinted}</div>
                    </div>
                    <div>
                      <Label>Успішність</Label>
                      <div className={`text-sm ${getSuccessRateColor(selectedModel.successRate)}`}>
                        {selectedModel.successRate}%
                      </div>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <Label>Основний матеріал</Label>
                    <div className="text-sm">{selectedModel.material}</div>
                  </div>

                  {selectedModel.additionalMaterials.length > 0 && (
                    <div>
                      <Label>Додаткові матеріали</Label>
                      <div className="space-y-1">
                        {selectedModel.additionalMaterials.map((material, index) => (
                          <div key={index} className="text-sm p-2 bg-muted rounded-md">
                            {material}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {selectedModel.customer && (
                    <div>
                      <Label>Замовник</Label>
                      <div className="text-sm">{selectedModel.customer}</div>
                    </div>
                  )}

                  <div>
                    <Label>Файли моделі ({selectedModel.files.length})</Label>
                    <div className="space-y-2">
                      {selectedModel.files.map((file, index) => (
                        <div key={index} className="flex items-center justify-between p-2 bg-muted rounded-md">
                          <div className="flex items-center gap-2">
                            <FileText className="h-4 w-4" />
                            <span className="text-sm">{file.name}</span>
                          </div>
                          <div className="text-xs text-muted-foreground">{file.size}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <Label>Нотатки</Label>
                    <div className="text-sm p-2 bg-muted rounded-md">{selectedModel.notes || "Немає нотаток"}</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}
