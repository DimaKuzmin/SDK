#include "stdafx.h"

#include "mu_model_light.h"
#include "mu_model_light_threads.h"
  
void	wait_mu_base		()
{
	run_mu_base();
	wait_mu_base_thread		();
}