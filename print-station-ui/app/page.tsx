"use client";

import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Printer,
  Package,
  Clock,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Play,
  Pause,
  Square,
} from "lucide-react";
import MaterialsTab from "@/components/materials-tab";
import ModelsTab from "@/components/models-tab";
import QueueTab from "@/components/queue-tab";
import AnalyticsTab from "@/components/analytics-tab";
import DashboardCharts from "@/components/dashboard-charts";
import StlViewer from "@/components/stl-viewer";
import {
  Dialog,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

export default function Dashboard() {
  const [currentPrint, setCurrentPrint] = useState({
    name: "Мініатюра дракона",
    progress: 67,
    timeLeft: "2г 15хв",
    status: "printing",
    stlUrl:
      // "https://files.printables.com/media/prints/3161/stls/123914_1f1d8ca1-252a-4770-846f-52f1208d193d/3dbenchy.stl",
      "https://files.printables.com/media/prints/1038186/stls/7873816_1285a7a0-0b3f-4ae8-aa5f-a328c7da114f_568f7dee-f1bc-4ecb-86f6-7aa3c16149a6/cutelittledragon.stl",
  });

  const stats = [
    {
      title: "Загальний час друку",
      value: "156г 23хв",
      description: "За цей місяць",
      icon: Clock,
      trend: "+12%",
    },
    {
      title: "Успішні друки",
      value: "47",
      description: "З 52 спроб",
      icon: CheckCircle,
      trend: "90.4%",
    },
    {
      title: "Витрачено пластику",
      value: "2.4 кг",
      description: "За цей місяць",
      icon: Package,
      trend: "+8%",
    },
    {
      title: "Економія коштів",
      value: "₴1,240",
      description: "Порівняно з покупкою",
      icon: TrendingUp,
      trend: "+15%",
    },
  ];

  const alerts = [
    { type: "warning", message: "PLA червоний закінчується (50г залишилось)" },
    { type: "info", message: "Час почистити сопло (45 годин роботи)" },
    { type: "success", message: "Калібрування столу завершено успішно" },
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Printer className="h-8 w-8 text-primary" />
              <h1 className="text-2xl font-bold">3D Принтер Контроль</h1>
            </div>
            <div className="flex items-center gap-4">
              <Badge
                variant={
                  currentPrint.status === "printing" ? "default" : "secondary"
                }
              >
                {currentPrint.status === "printing" ? "Друкує" : "Готовий"}
              </Badge>
              <div className="text-sm text-muted-foreground">
                Останнє оновлення: щойно
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-12 gap-6">
          {/* Current Print Status */}
          {currentPrint.status === "printing" && (
            <Card className="mb-6 col-span-12 lg:col-span-8">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <Play className="h-5 w-5 text-green-500" />
                      Поточний друк: {currentPrint.name}
                    </CardTitle>
                    <CardDescription>
                      Залишилось: {currentPrint.timeLeft}
                    </CardDescription>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm">
                      <Pause className="h-4 w-4" />
                    </Button>
                    <Button variant="outline" size="sm">
                      <Square className="h-4 w-4" />
                    </Button>

                    <Dialog>
                      <DialogTrigger asChild>
                        <Button variant="outline" size="sm">
                          Перегляд
                        </Button>
                      </DialogTrigger>
                      <DialogContent className="max-w-5xl">
                        <DialogHeader>
                          <DialogTitle>
                            Превʼю моделі: {currentPrint.name}
                          </DialogTitle>
                        </DialogHeader>
                        <StlViewer url={currentPrint.stlUrl} />
                      </DialogContent>
                    </Dialog>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Прогрес</span>
                    <span>{currentPrint.progress}%</span>
                  </div>
                  <Progress value={currentPrint.progress} className="h-2" />
                </div>
              </CardContent>
            </Card>
          )}

          {/* Alerts */}
          {alerts.length > 0 && (
            <Card className="mb-6 col-span-12 lg:col-span-4">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5" />
                  Сповіщення
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {alerts.map((alert, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-2 p-2 rounded-lg bg-muted/50"
                    >
                      <div
                        className={`w-2 h-2 rounded-full ${
                          alert.type === "warning"
                            ? "bg-yellow-500"
                            : alert.type === "info"
                            ? "bg-blue-500"
                            : "bg-green-500"
                        }`}
                      />
                      <span className="text-sm">{alert.message}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
          {stats.map((stat, index) => (
            <Card key={index}>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">
                  {stat.title}
                </CardTitle>
                <stat.icon className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stat.value}</div>
                <p className="text-xs text-muted-foreground">
                  {stat.description}
                </p>
                <div className="text-xs text-green-600 mt-1">{stat.trend}</div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Dashboard Charts */}
        <DashboardCharts />

        {/* Main Tabs */}
        <Tabs defaultValue="materials" className="space-y-4">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="materials">Матеріали</TabsTrigger>
            <TabsTrigger value="models">Моделі</TabsTrigger>
            <TabsTrigger value="queue">Черга</TabsTrigger>
            <TabsTrigger value="analytics">Аналітика</TabsTrigger>
          </TabsList>

          <TabsContent value="materials">
            <MaterialsTab />
          </TabsContent>

          <TabsContent value="models">
            <ModelsTab />
          </TabsContent>

          <TabsContent value="queue">
            <QueueTab />
          </TabsContent>

          <TabsContent value="analytics">
            <AnalyticsTab />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
