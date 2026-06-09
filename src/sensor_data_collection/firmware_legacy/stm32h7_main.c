/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2025 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "dma.h"
#include "i2c.h"
#include "memorymap.h"
#include "spi.h"
#include "stm32h7xx_hal_gpio.h"
#include "tim.h"
#include "usb_device.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "sensor_qmc6309.h"
#include "tca9548.h"
#include "icm42670.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "usbd_cdc_if.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define TCA9548_ADDR1 0x70 // I2C1上的TCA9548地址
#define TCA9548_ADDR2 0x70 // I2C2上的TCA9548地址（如不同可改）
#define VOLTAGE_MODE 1
#define CURRENT_MODE 2
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
uint8_t usb_rx_buffer[256];
uint8_t usb_tx_buffer[256];
volatile uint8_t usb_rx_flag = 0;
uint32_t usb_rx_length = 0;

uint32_t last_timer_tick = 0;
uint8_t sg_start = 0;

volatile uint8_t is_configured = 0;
uint8_t config_mode = 0;
uint16_t config_sensor_bitmap = 0;
uint32_t config_settling_time_us = 0;
uint32_t config_cycle_time_us = 0;

volatile uint8_t sg_sync_triggered = 0;
uint32_t sync_timestamp_us = 0;
uint16_t current_cycle_id = 0;

uint64_t last_sync_int = 0;
uint64_t last_cdc_tx_time = 0;
uint64_t last_cdc_rx_time = 0;

icm42670_raw_t imu;
icm42670_t icm = {
    .hspi = &hspi1,
    .cs_port = SPI1_CS_GPIO_Port,
    .cs_pin = SPI1_CS_Pin};

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MPU_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
// USB CDC发送字符串
void USB_Send_String(char *str)
{
    CDC_Transmit_FS((uint8_t *)str, strlen(str));
}

// printf 重定向到 USB CDC
int _write(int file, char *ptr, int len)
{
    // HAL_UART_Transmit(&huart1, (uint8_t *)ptr, len, 1000);
    CDC_Transmit_FS((uint8_t *)ptr, len);
    return len;
}

// 获取微秒级时间戳
uint32_t get_us(void)
{
    return __HAL_TIM_GET_COUNTER(&htim5);
}

// 获取64位微秒级时间戳
uint64_t get_us_64(void)
{
    static uint32_t last_us = 0;
    static uint64_t us_64 = 0;
    
    uint32_t primask = __get_PRIMASK();
    __disable_irq();

    uint32_t now = __HAL_TIM_GET_COUNTER(&htim5);
    if (now < last_us) {
        us_64 += ((1ULL << 32) - last_us + now);
    } else {
        us_64 += (now - last_us);
    }
    last_us = now;

    __set_PRIMASK(primask);
    return us_64;
}

// 微秒级阻塞延时
void delay_us(uint32_t us)
{
    uint32_t start = get_us();
    while ((get_us() - start) < us);
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MPU Configuration--------------------------------------------------------*/
  MPU_Config();

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_I2C1_Init();
  MX_I2C2_Init();
  MX_SPI1_Init();
  MX_USB_DEVICE_Init();
  MX_TIM5_Init();
  /* USER CODE BEGIN 2 */

    HAL_TIM_Base_Start(&htim5);

    HAL_GPIO_WritePin(LEDR_GPIO_Port, LEDR_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(LEDG_GPIO_Port, LEDG_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(LEDB_GPIO_Port, LEDB_Pin, GPIO_PIN_SET);

    HAL_Delay(100);


    Sensor_QMC6309_Init_All();
    if (ICM42670_Init(&icm) != HAL_OK)
    {
        printf("ICM init FAILED\r\n");
        HAL_GPIO_WritePin(LEDR_GPIO_Port, LEDR_Pin, GPIO_PIN_RESET); // 红灯亮
        // Error_Handler();
    }
    else
    {
        printf("ICM init OK (WHO=0x67)\r\n");
    }

    last_timer_tick = get_us();

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
    // static uint64_t last_toggle_time = 0;
    while (1)
    {
        // get_us_64(); // 维护64位计时器

        uint64_t now = get_us_64();
        
        if(now - last_sync_int >= 10000) {
            HAL_GPIO_WritePin(LEDG_GPIO_Port, LEDG_Pin, GPIO_PIN_SET); 
        }
        else 
        {
            HAL_GPIO_WritePin(LEDG_GPIO_Port, LEDG_Pin, GPIO_PIN_RESET); // 同步信号SG_SYNC在10ms内未触发则绿灯闪亮
        }

        if(is_configured && (now - last_cdc_tx_time > 10000)) {
            HAL_GPIO_WritePin(LEDB_GPIO_Port, LEDB_Pin, GPIO_PIN_RESET); // 配置完成蓝灯常亮
        } else {
            HAL_GPIO_WritePin(LEDB_GPIO_Port, LEDB_Pin, GPIO_PIN_SET); // 未配置蓝灯熄灭
        }

        if (usb_rx_flag)
        {
            if (usb_rx_length == 12 && usb_rx_buffer[0] == 0xAA && usb_rx_buffer[1] == 0x55)
            {
                uint8_t version = usb_rx_buffer[2];
                uint8_t mode = usb_rx_buffer[3];
                uint16_t bitmap = (usb_rx_buffer[4] << 8) | usb_rx_buffer[5];
                uint16_t settle = (usb_rx_buffer[6] << 8) | usb_rx_buffer[7];
                uint32_t cycle = ((uint32_t)usb_rx_buffer[8] << 24) | ((uint32_t)usb_rx_buffer[9] << 16) | ((uint32_t)usb_rx_buffer[10] << 8) | usb_rx_buffer[11];
                
                if (mode == VOLTAGE_MODE || mode == CURRENT_MODE)
                {
                    config_mode = mode;
                    config_sensor_bitmap = bitmap;
                    config_settling_time_us = (uint32_t)settle * 10;
                    config_cycle_time_us = cycle * 10;
                    is_configured = 1;
                    
                    uint8_t reply[3] = {0xAA, 0x55, 0x00};
                    CDC_Transmit_FS(reply, 3);
                }
                else
                {
                    uint8_t reply[3] = {0xAA, 0x55, 0x01}; // 参数错误
                    CDC_Transmit_FS(reply, 3);
                }
            }
            else
            {
                if (usb_rx_length > 0 && usb_rx_buffer[0] == 0xAA && usb_rx_buffer[1] == 0x55) {
                    uint8_t reply[3] = {0xAA, 0x55, 0xFF}; // 未知错误
                    CDC_Transmit_FS(reply, 3);
                }
            }
            usb_rx_flag = 0;
        }

        if (sg_sync_triggered)
        {
            sg_sync_triggered = 0;
            
            uint32_t slot_num = 1;
            if(config_mode == VOLTAGE_MODE) {
                slot_num = 4;
            } else if (config_mode == CURRENT_MODE) {
                slot_num = 3;
            }
            if (slot_num > 0 && config_cycle_time_us > 0)
            {
                uint32_t slot_interval_us = config_cycle_time_us / slot_num;
                
                for (uint8_t slot = 0; slot < slot_num; slot++)
                {
                    uint32_t slot_start_target = sync_timestamp_us + slot * slot_interval_us;
                    
                    // 等待直到 slot 开始 (处理环绕)
                    while ((int32_t)(get_us() - slot_start_target) < 0) { }
                    
                    // 延时 settling time
                    delay_us(config_settling_time_us);
                    
                    uint64_t ts_64 = get_us_64();
                    
                    uint8_t tx_buf[256];
                    int tlen = 0;
                    tx_buf[tlen++] = 0xAA;
                    tx_buf[tlen++] = 0x55;
                    tx_buf[tlen++] = 0x01; // Version
                    tx_buf[tlen++] = (current_cycle_id >> 8) & 0xFF;
                    tx_buf[tlen++] = current_cycle_id & 0xFF;
                    tx_buf[tlen++] = slot;
                    tx_buf[tlen++] = (config_sensor_bitmap >> 8) & 0xFF;
                    tx_buf[tlen++] = config_sensor_bitmap & 0xFF;
                    
                    for (int i=7; i>=0; i--) {
                        tx_buf[tlen++] = (ts_64 >> (i*8)) & 0xFF;
                    }
                    
                    for (int i=0; i<12; i++) {
                        if (config_sensor_bitmap & (1<<i)) {
                            int16_t x=0, y=0, z=0;
                            Sensor_QMC6309_ReadRawData(i, &x, &y, &z);
                            tx_buf[tlen++] = i + 1; // Sensor ID: 1~12
                            
                            int32_t x32 = x; int32_t y32 = y; int32_t z32 = z;
                            tx_buf[tlen++] = (x32 >> 24) & 0xFF;
                            tx_buf[tlen++] = (x32 >> 16) & 0xFF;
                            tx_buf[tlen++] = (x32 >> 8) & 0xFF;
                            tx_buf[tlen++] = x32 & 0xFF;
                            
                            tx_buf[tlen++] = (y32 >> 24) & 0xFF;
                            tx_buf[tlen++] = (y32 >> 16) & 0xFF;
                            tx_buf[tlen++] = (y32 >> 8) & 0xFF;
                            tx_buf[tlen++] = y32 & 0xFF;
                            
                            tx_buf[tlen++] = (z32 >> 24) & 0xFF;
                            tx_buf[tlen++] = (z32 >> 16) & 0xFF;
                            tx_buf[tlen++] = (z32 >> 8) & 0xFF;
                            tx_buf[tlen++] = z32 & 0xFF;
                        }
                    }
                    
                    tx_buf[tlen++] = (slot == slot_num - 1) ? 0x01 : 0x00;
                    
                    // 等待CDC不忙时再发
                    while(CDC_Transmit_FS(tx_buf, tlen) == USBD_BUSY) {
                         // 防止卡死，可以稍等
                         delay_us(100);
                    }
                    last_cdc_tx_time = get_us_64();
                }
                current_cycle_id++;
            }
        }
    }
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 5;
  RCC_OscInitStruct.PLL.PLLN = 48;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 5;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_2;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV4;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
    uint64_t now = get_us_64();
    // SG_SYNC引脚中断
    if (GPIO_Pin == SG_SYNC_Pin && (now - last_sync_int) > 1000) // 简单去抖，至少间隔1ms
    {
        if (is_configured)
        {
            sg_sync_triggered = 1;
            sync_timestamp_us = get_us();
        }
        last_sync_int = now;
    }
}

/* USER CODE END 4 */

 /* MPU Configuration */

void MPU_Config(void)
{
  MPU_Region_InitTypeDef MPU_InitStruct = {0};

  /* Disables the MPU */
  HAL_MPU_Disable();

  /** Initializes and configures the Region and the memory to be protected
  */
  MPU_InitStruct.Enable = MPU_REGION_ENABLE;
  MPU_InitStruct.Number = MPU_REGION_NUMBER0;
  MPU_InitStruct.BaseAddress = 0x0;
  MPU_InitStruct.Size = MPU_REGION_SIZE_4GB;
  MPU_InitStruct.SubRegionDisable = 0x87;
  MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
  MPU_InitStruct.AccessPermission = MPU_REGION_NO_ACCESS;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
  MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
  MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;

  HAL_MPU_ConfigRegion(&MPU_InitStruct);
  /* Enables the MPU */
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);

}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
    __disable_irq();
    // 持续亮红灯
    HAL_GPIO_WritePin(LEDR_GPIO_Port, LEDR_Pin, GPIO_PIN_RESET);
    while (1)
    {
        HAL_Delay(1000);
    }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
    /* User can add his own implementation to report the file name and line number,
       ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
