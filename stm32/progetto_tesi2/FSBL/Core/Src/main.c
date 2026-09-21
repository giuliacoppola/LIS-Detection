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
#include "app_x-cube-ai.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
CACHEAXI_HandleTypeDef hcacheaxi;

UART_HandleTypeDef hlpuart1;

/* USER CODE BEGIN PV */

LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(Default)
uint8_t rx_data;
char tx_buffer[50];
#define NUM_FEATURES 42  // 42 valori float da Mediapipe
uint8_t rx_bytes[4];
float input_buffer[NUM_FEATURES];
volatile uint8_t float_byte_count = 0;
volatile uint8_t feature_count = 0;
volatile uint8_t start_inference = 0;

static char rx_buffer[512];
static uint16_t rx_index = 0;
uint8_t flag_feature_received = 0;

//extern uint8_t *buffer_in;
//extern uint8_t *buffer_out;



/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_CACHEAXI_Init(void);
static void MX_LPUART1_UART_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

int parse_int8_buffer(const char *input, int8_t *output, int max_len)
{
	if (input == NULL || output == NULL) {
		return 0;
	}

	int count = 0;
	char buffer[512]; // copia temporanea della stringa (dipende dalla lunghezza massima)
	strncpy(buffer, input, sizeof(buffer) - 1);
	buffer[sizeof(buffer) - 1] = '\0';

	char *token = strtok(buffer, ":");
	while (token != NULL && count < max_len) {
		long value = strtol(token, NULL, 10);

		//clamp dei valori validi per int8_t
		if (value < -127) value = -127;
		else if (value > 128) value = 128;

		output[count++] = (int8_t)value;




		token = strtok(NULL, ":");
	}


	return count; // ritorna quanti numeri ha trovato
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

	/* Enable the CPU Cache */

	/* Enable I-Cache---------------------------------------------------------*/
	SCB_EnableICache();

	/* Enable D-Cache---------------------------------------------------------*/
	SCB_EnableDCache();

	/* MCU Configuration--------------------------------------------------------*/
	HAL_Init();

	/* USER CODE BEGIN Init */

	/* USER CODE END Init */

	/* Configure the system clock */
	SystemClock_Config();

	/* USER CODE BEGIN SysInit */

	/* USER CODE END SysInit */

	/* Initialize all configured peripherals */
	MX_GPIO_Init();
	MX_CACHEAXI_Init();
	MX_LPUART1_UART_Init();
	MX_X_CUBE_AI_Init();
	/* USER CODE BEGIN 2 */

	//HAL_UART_Transmit(&hlpuart1, (uint8_t*)"STM ready\r\n", 11, HAL_MAX_DELAY); // messaggio di test
	HAL_UART_Receive_IT(&hlpuart1, &rx_data, 1);  // attiva la ricezione UART

	/* USER CODE END 2 */

	/* Infinite loop */
	/* USER CODE BEGIN WHILE */
	while (1)
	{
		/* USER CODE END WHILE */

		/* USER CODE BEGIN 3 */


		HAL_GPIO_TogglePin(GPIOG, GPIO_PIN_8);
		HAL_Delay(100);


		// Se abbiamo ricevuto 42 feature dal PC (flag alzato)
		if (flag_feature_received)
		{
			flag_feature_received = 0;

			//HAL_UART_Transmit(&hlpuart1, (uint8_t*)rx_buffer, strlen(rx_buffer), HAL_MAX_DELAY);

			int8_t *buffer_in;
			int8_t *buffer_out;

			LL_ATON_RT_RetValues_t ll_aton_rt_ret = LL_ATON_RT_DONE;
			const LL_Buffer_InfoTypeDef * ibuffersInfos = NN_Interface_Default.input_buffers_info();
			const LL_Buffer_InfoTypeDef * obuffersInfos = NN_Interface_Default.output_buffers_info();
			buffer_in = (int8_t *)LL_Buffer_addr_start(&ibuffersInfos[0]);
			buffer_out = (int8_t *)LL_Buffer_addr_start(&obuffersInfos[0]);

			//int8_t feature_buffer[NUM_FEATURES];
			int n = parse_int8_buffer(rx_buffer, buffer_in, NUM_FEATURES);

			//snprintf(tx_buffer, sizeof(tx_buffer), "n: %d\r\n", n);
			//HAL_UART_Transmit(&hlpuart1, (uint8_t*)tx_buffer, strlen(tx_buffer), HAL_MAX_DELAY);


			//MX_X_CUBE_AI_Process();

			LL_ATON_RT_RuntimeInit();

			LL_ATON_RT_Init_Network(&NN_Instance_Default);  // Initialize passed network instance object

			do {
				/* Execute first/next step */
				ll_aton_rt_ret = LL_ATON_RT_RunEpochBlock(&NN_Instance_Default);
				/* Wait for next event */
				if (ll_aton_rt_ret == LL_ATON_RT_WFE) {
					LL_ATON_OSAL_WFE();
				}
			} while (ll_aton_rt_ret != LL_ATON_RT_DONE);

			/* Post-process the output buffer */
			/* Invalidate the associated CPU cache region if requested */
			//_post_process(buffer_out);


			LL_ATON_RT_DeInit_Network(&NN_Instance_Default);
			LL_ATON_RT_RuntimeDeInit();




			int num_values = LL_Buffer_len(&obuffersInfos[0]);


			//snprintf(tx_buffer, sizeof(tx_buffer), "Output bytes: %d\r\n", num_values);
			//HAL_UART_Transmit(&hlpuart1, (uint8_t*)tx_buffer, strlen(tx_buffer), HAL_MAX_DELAY);

			// Trasmetti tutti i valori int8
			for (int i = 0; i < num_values; i++) {
			    char temp[16];
			    snprintf(temp, sizeof(temp), "%d:", buffer_in[i]);
			    HAL_UART_Transmit(&hlpuart1, (uint8_t*)temp, strlen(temp), HAL_MAX_DELAY);
			}
			HAL_UART_Transmit(&hlpuart1, (uint8_t*)"\r\n", 2, HAL_MAX_DELAY);


			// TEST: rispondi sempre con una lettera fissa per verificare la comunicazione
			//char pred = 'A';

			//snprintf(tx_buffer, sizeof(tx_buffer), "Predizione: %c\r\n", pred);
			//HAL_UART_Transmit(&hlpuart1, (uint8_t*)tx_buffer, strlen(tx_buffer), HAL_MAX_DELAY);
		}






	}

	/* USER CODE END 3 */
}
/* USER CODE BEGIN CLK 1 */
/* USER CODE END CLK 1 */

/**
 * @brief System Clock Configuration
 * @retval None
 */
void SystemClock_Config(void)
{
	RCC_OscInitTypeDef RCC_OscInitStruct = {0};
	RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

	/** Configure the System Power Supply
	 */
	if (HAL_PWREx_ConfigSupply(PWR_EXTERNAL_SOURCE_SUPPLY) != HAL_OK)
	{
		Error_Handler();
	}

	/* Enable HSI */
	RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
	RCC_OscInitStruct.HSIState = RCC_HSI_ON;
	RCC_OscInitStruct.HSIDiv = RCC_HSI_DIV1;
	RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
	RCC_OscInitStruct.PLL1.PLLState = RCC_PLL_NONE;
	RCC_OscInitStruct.PLL2.PLLState = RCC_PLL_NONE;
	RCC_OscInitStruct.PLL3.PLLState = RCC_PLL_NONE;
	RCC_OscInitStruct.PLL4.PLLState = RCC_PLL_NONE;
	if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
	{
		Error_Handler();
	}

	/** Get current CPU/System buses clocks configuration and if necessary switch
 to intermediate HSI clock to ensure target clock can be set
	 */
	HAL_RCC_GetClockConfig(&RCC_ClkInitStruct);
	if ((RCC_ClkInitStruct.CPUCLKSource == RCC_CPUCLKSOURCE_IC1) ||
			(RCC_ClkInitStruct.SYSCLKSource == RCC_SYSCLKSOURCE_IC2_IC6_IC11))
	{
		RCC_ClkInitStruct.ClockType = (RCC_CLOCKTYPE_CPUCLK | RCC_CLOCKTYPE_SYSCLK);
		RCC_ClkInitStruct.CPUCLKSource = RCC_CPUCLKSOURCE_HSI;
		RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
		if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct) != HAL_OK)
		{
			/* Initialization Error */
			Error_Handler();
		}
	}

	/** Initializes the RCC Oscillators according to the specified parameters
	 * in the RCC_OscInitTypeDef structure.
	 */
	RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_NONE;
	RCC_OscInitStruct.PLL1.PLLState = RCC_PLL_ON;
	RCC_OscInitStruct.PLL1.PLLSource = RCC_PLLSOURCE_HSI;
	RCC_OscInitStruct.PLL1.PLLM = 2;
	RCC_OscInitStruct.PLL1.PLLN = 75;
	RCC_OscInitStruct.PLL1.PLLFractional = 0;
	RCC_OscInitStruct.PLL1.PLLP1 = 1;
	RCC_OscInitStruct.PLL1.PLLP2 = 1;
	RCC_OscInitStruct.PLL2.PLLState = RCC_PLL_ON;
	RCC_OscInitStruct.PLL2.PLLSource = RCC_PLLSOURCE_HSI;
	RCC_OscInitStruct.PLL2.PLLM = 4;
	RCC_OscInitStruct.PLL2.PLLN = 75;
	RCC_OscInitStruct.PLL2.PLLFractional = 0;
	RCC_OscInitStruct.PLL2.PLLP1 = 2;
	RCC_OscInitStruct.PLL2.PLLP2 = 1;
	RCC_OscInitStruct.PLL3.PLLState = RCC_PLL_NONE;
	RCC_OscInitStruct.PLL4.PLLState = RCC_PLL_NONE;
	if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
	{
		Error_Handler();
	}

	/** Initializes the CPU, AHB and APB buses clocks
	 */
	RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_CPUCLK|RCC_CLOCKTYPE_HCLK
			|RCC_CLOCKTYPE_SYSCLK|RCC_CLOCKTYPE_PCLK1
			|RCC_CLOCKTYPE_PCLK2|RCC_CLOCKTYPE_PCLK5
			|RCC_CLOCKTYPE_PCLK4;
	RCC_ClkInitStruct.CPUCLKSource = RCC_CPUCLKSOURCE_IC1;
	RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_IC2_IC6_IC11;
	RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
	RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV1;
	RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV1;
	RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV1;
	RCC_ClkInitStruct.APB5CLKDivider = RCC_APB5_DIV1;
	RCC_ClkInitStruct.IC1Selection.ClockSelection = RCC_ICCLKSOURCE_PLL2;
	RCC_ClkInitStruct.IC1Selection.ClockDivider = 1;
	RCC_ClkInitStruct.IC2Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
	RCC_ClkInitStruct.IC2Selection.ClockDivider = 6;
	RCC_ClkInitStruct.IC6Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
	RCC_ClkInitStruct.IC6Selection.ClockDivider = 3;
	RCC_ClkInitStruct.IC11Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
	RCC_ClkInitStruct.IC11Selection.ClockDivider = 3;

	if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct) != HAL_OK)
	{
		Error_Handler();
	}
}

/**
 * @brief CACHEAXI Initialization Function
 * @param None
 * @retval None
 */
static void MX_CACHEAXI_Init(void)
{

	/* USER CODE BEGIN CACHEAXI_Init 0 */

	/* USER CODE END CACHEAXI_Init 0 */

	/* USER CODE BEGIN CACHEAXI_Init 1 */

	/* USER CODE END CACHEAXI_Init 1 */
	hcacheaxi.Instance = CACHEAXI;
	if (HAL_CACHEAXI_Init(&hcacheaxi) != HAL_OK)
	{
		Error_Handler();
	}
	/* USER CODE BEGIN CACHEAXI_Init 2 */

	/* USER CODE END CACHEAXI_Init 2 */

}

/**
 * @brief LPUART1 Initialization Function
 * @param None
 * @retval None
 */
static void MX_LPUART1_UART_Init(void)
{

	/* USER CODE BEGIN LPUART1_Init 0 */

	/* USER CODE END LPUART1_Init 0 */

	/* USER CODE BEGIN LPUART1_Init 1 */

	/* USER CODE END LPUART1_Init 1 */
	hlpuart1.Instance = LPUART1;
	hlpuart1.Init.BaudRate = 115200;
	hlpuart1.Init.WordLength = UART_WORDLENGTH_8B;
	hlpuart1.Init.StopBits = UART_STOPBITS_1;
	hlpuart1.Init.Parity = UART_PARITY_NONE;
	hlpuart1.Init.Mode = UART_MODE_TX_RX;
	hlpuart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
	hlpuart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
	hlpuart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
	hlpuart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
	hlpuart1.FifoMode = UART_FIFOMODE_DISABLE;
	if (HAL_UART_Init(&hlpuart1) != HAL_OK)
	{
		Error_Handler();
	}
	if (HAL_UARTEx_SetTxFifoThreshold(&hlpuart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
	{
		Error_Handler();
	}
	if (HAL_UARTEx_SetRxFifoThreshold(&hlpuart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
	{
		Error_Handler();
	}
	if (HAL_UARTEx_DisableFifoMode(&hlpuart1) != HAL_OK)
	{
		Error_Handler();
	}
	/* USER CODE BEGIN LPUART1_Init 2 */

	/* USER CODE END LPUART1_Init 2 */

}

/**
 * @brief GPIO Initialization Function
 * @param None
 * @retval None
 */
static void MX_GPIO_Init(void)
{
	GPIO_InitTypeDef GPIO_InitStruct = {0};
	/* USER CODE BEGIN MX_GPIO_Init_1 */

	/* USER CODE END MX_GPIO_Init_1 */

	/* GPIO Ports Clock Enable */
	__HAL_RCC_GPIOC_CLK_ENABLE();
	__HAL_RCC_GPIOE_CLK_ENABLE();
	__HAL_RCC_GPIOH_CLK_ENABLE();
	__HAL_RCC_GPIOB_CLK_ENABLE();
	__HAL_RCC_GPIOA_CLK_ENABLE();
	__HAL_RCC_GPIOG_CLK_ENABLE();

	/*Configure GPIO pin Output Level */
	HAL_GPIO_WritePin(GPIOG, LED3_Pin|LED2_Pin|LED1_Pin, GPIO_PIN_RESET);

	/*Configure GPIO pin : I2C1_SDA_Pin */
	GPIO_InitStruct.Pin = I2C1_SDA_Pin;
	GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
	GPIO_InitStruct.Alternate = GPIO_AF4_I2C1;
	HAL_GPIO_Init(I2C1_SDA_GPIO_Port, &GPIO_InitStruct);

	/*Configure GPIO pin : I2CA_SCL_Pin */
	GPIO_InitStruct.Pin = I2CA_SCL_Pin;
	GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
	GPIO_InitStruct.Alternate = GPIO_AF4_I2C1;
	HAL_GPIO_Init(I2CA_SCL_GPIO_Port, &GPIO_InitStruct);

	/*Configure GPIO pins : I2C2_SDA_Pin I2C2_SCL_Pin */
	GPIO_InitStruct.Pin = I2C2_SDA_Pin|I2C2_SCL_Pin;
	GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
	GPIO_InitStruct.Alternate = GPIO_AF4_I2C2;
	HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

	/*Configure GPIO pins : PA10 UCPD1_VSENSE_Pin */
	GPIO_InitStruct.Pin = GPIO_PIN_10|UCPD1_VSENSE_Pin;
	GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

	/*Configure GPIO pins : LED3_Pin LED2_Pin LED1_Pin */
	GPIO_InitStruct.Pin = LED3_Pin|LED2_Pin|LED1_Pin;
	GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
	HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

	/* USER CODE BEGIN MX_GPIO_Init_2 */

	/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
	if (huart->Instance == LPUART1)
	{

		if (rx_data == '\n') {
			rx_buffer[rx_index] = '\0'; // chiudo la stringa

			// rispedisco la stringa ricevuta
			//HAL_UART_Transmit(&hlpuart1, (uint8_t*)rx_buffer, strlen(rx_buffer), HAL_MAX_DELAY);
			flag_feature_received = 1;

			// Azzero l’indice per la prossima stringa
			rx_index = 0;
		}
		else {
			if (rx_index < sizeof(rx_buffer) - 1) {
				rx_buffer[rx_index++] = rx_data;
			}
		}

		// Riattiva la ricezione del prossimo carattere
		HAL_UART_Receive_IT(&hlpuart1, &rx_data, 1);
	}
}


/*

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == LPUART1)
    {

    	if (rx_data == '\n'){
    		HAL_UART_Transmit(&hlpuart1, (uint8_t*)rx_bytes, strlen(rx_bytes), HAL_MAX_DELAY);
    		float_byte_count = 0;
    	} else {
    		rx_bytes[float_byte_count++] = rx_data;
    	}




    	// Salva il byte nel buffer temporaneo
        rx_bytes[float_byte_count++] = rx_data;

        // Se abbiamo ricevuto 4 byte → ricomponi il float
        if (float_byte_count == 4)
        {
            float value;
            memcpy(&value, rx_bytes, 4);   // ricostruisci il float
            input_buffer[feature_count++] = value;
            float_byte_count = 0;

            // Se abbiamo ricevuto 42 feature, alza il flag
            if (feature_count == NUM_FEATURES)
            {
                start_inference = 1;
                feature_count = 0;  // reset per il prossimo batch
            }




        // Riabilita la ricezione per il prossimo byte
        HAL_UART_Receive_IT(&hlpuart1, &rx_data, 1);
    }
}

 */





/* USER CODE END 4 */

/**
 * @brief  This function is executed in case of error occurrence.
 * @retval None
 */
void Error_Handler(void)
{
	/* USER CODE BEGIN Error_Handler_Debug */
	/* User can add his own implementation to report the HAL error return state */
	__disable_irq();
	while (1)
	{
	}
	/* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
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
