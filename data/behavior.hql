create temporary table nlp.expect_profile as
select
  a.expect_id, a.geek_id, a.l3_name, a.city, 
  a.gender, a.degree, a.fresh_graduate, a.apply_status, a.completion,
  b.cv
from
  nlp.expect_profile_category a
join
  nlp.geek_cv_raw b
on
  a.geek_id = b.geek_id
where
  a.l1_name = '技术';

create temporary table nlp.job_profile as
select
  a.job_id, a.boss_id, a.position, a.city, a.degree, a.experience, a.area_business_name,
  a.boss_title, a.is_hr, a.stage,
  b.jd
from
  nlp.job_profile_category a
join
  nlp.job_desc_raw b
on
  a.job_id = b.job_id
where
  a.l1_code = 100000
and
  a.is_hr = 0;

create temporary table nlp.qyh_boss_add_friend as
  select
    actionp geek_id, actionp2 job_id, actionp3 expect_id, min(ds) ds, 2 action
  from
    dw_bosszp.bg_action a
  join
    nlp.job_profile b
  on
    a.actionp2 = b.job_id
  where
    ds >= '2018-10-01' and ds < '2018-11-22'
  and
    action = 'detail-geek-addfriend'
  and
    bg = 1
  group by
    actionp, actionp2, actionp3
  distribute by rand()
  sort by rand();

create temporary table nlp.qyh_boss_interview_accept_tmp as
  select
    actionp2 job_id, uid geek_id, min(ds) ds, 3 action
  from
    dw_bosszp.bg_action
  where
    ds >= '2018-10-01' and ds < '2018-11-22'
  and
    action = 'chat-interview-accept'
  and
    bg = 0
  group by
    uid, actionp2;

create temporary table nlp.qyh_boss_interview_accept as
  select
    b.job_id job_id, b.expect_id expect_id, a.ds ds, a.action action
  from
    nlp.qyh_boss_interview_accept_tmp a
  join
    nlp.qyh_boss_add_friend b
  on
    a.geek_id = b.geek_id and a.job_id = b.job_id
  distribute by rand()
  sort by rand()
  limit 100000;

insert overwrite table nlp.qyh_boss_add_friend
  select
    geek_id, job_id, expect_id, ds, action
  from
    nlp.qyh_boss_add_friend
  limit 100000;

create temporary table nlp.qyh_boss_view as
  select
    actionp3 job_id, actionp2 expect_id, min(ds) ds, 1 action
  from
    dw_bosszp.bg_action a
  join
    nlp.job_profile b
  on
    a.actionp3 = b.job_id    
  where
    ds >= '2018-10-01' and ds < '2018-11-22'
  and
    action = 'detail-geek'
  and
    bg = 1
  group by
    actionp2, actionp3
  distribute by rand()
  sort by rand()
  limit 100000;

create temporary table nlp.qyh_boss_list as
  select
    jobid job_id, expectid expect_id, min(ds) ds, 0 action
  from
    dw_bosszp.bg_list_action a
  join
    nlp.job_profile b
  on
    a.jobid = b.job_id
  where
    ds >= '2018-10-01' and ds < '2018-11-22'
  and
    bg = 1
  and
    pagenumber = 1
  group by
    jobid, expectid
  distribute by rand()
  sort by rand()
  limit 100000;

create temporary table nlp.qyh_view_detail_add_interview_sample as
  select 
    job_id, expect_id, min(ds) ds, max(action) action
  from (
    select
      job_id, expect_id, ds, action
    from
      nlp.qyh_boss_interview_accept
    union all
    select
      job_id, expect_id, ds, action
    from
      nlp.qyh_boss_view
    union all
    select
      job_id, expect_id, ds, action
    from
      nlp.qyh_boss_add_friend
    union all
    select
      job_id, expect_id, ds, action
    from
      nlp.qyh_boss_list
  ) a
  group by
    job_id, expect_id;

create temporary table nlp.qyh_interview_sample_train as
  select
    expect_id, job_id, action
  from 
    nlp.qyh_view_detail_add_interview_sample
  where
    ds < '2018-11-15'
  distribute by rand()
  sort by rand();

create temporary table nlp.qyh_interview_sample_test as
  select
    expect_id, job_id, action
  from 
    nlp.qyh_view_detail_add_interview_sample
  where
    ds >= '2018-11-15'
  distribute by rand()
  sort by rand();

-- create temporary table nlp.qyh_interview_sample_per_job as
-- select
--   job_id, expect_id, action
-- from (
--   select
--     job_id, expect_id, action, row_number() over (partition by job_id, action order by ds asc) num
--   from
--     nlp.qyh_interview_sample_test
-- ) a
-- where
--   a.num = 1;

insert overwrite local directory 'train'
  select
    a.expect_id, a.job_id, action
  from
    nlp.qyh_interview_sample_train a
  join
    nlp.expect_profile b
  on
    a.expect_id = b.expect_id
  join
    nlp.job_profile c
  on
    a.job_id = c.job_id;

insert overwrite local directory 'test'
  select
    a.expect_id, a.job_id, action
  from
    nlp.qyh_interview_sample_test a
  join
    nlp.expect_profile b
  on
    a.expect_id = b.expect_id
  join
    nlp.job_profile c
  on
    a.job_id = c.job_id;

insert overwrite local directory 'expect_profile'
  select
    distinct(a.expect_id), 
    a.geek_id, a.l3_name, a.city, 
    a.gender, a.degree, a.fresh_graduate, 
    a.apply_status, a.completion, a.cv
  from
    nlp.expect_profile a
  join
    nlp.qyh_view_detail_add_interview_sample b
  on
    a.expect_id = b.expect_id;

insert overwrite local directory 'job_profile'
  select
    distinct(a.job_id), 
    a.boss_id, a.position, a.city, a.degree, 
    a.experience, a.area_business_name, 
    a.boss_title, a.is_hr, a.stage, a.jd
  from
    nlp.job_profile a
  join
    nlp.qyh_view_detail_add_interview_sample b
  on
    a.job_id = b.job_id;